import torch
import random
import numpy as np
import logging
import copy
import os
import json

from tqdm import tqdm
from DensePhrases.densephrases.utils.single_utils import load_encoder
from DensePhrases.densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from DensePhrases.densephrases.utils.eval_utils import (
    drqa_exact_match_score, drqa_regex_match_score
)
from DensePhrases.eval_phrase_retrieval import (
    evaluate_results,
    embed_all_query
)
from DensePhrases.densephrases import Options
from cross_encoder import CrossEncoder
from torch.nn.functional import softmax
from numpy.random import default_rng
rng = default_rng()

from transformers import get_linear_schedule_with_warmup

from torch.optim import SGD

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_top_phrases_query_vec(mips, questions, query_vecs, args):
    # Search
    search_fn = mips.search

    outs = search_fn(
        query_vecs.cpu().detach().numpy(),
        q_texts=questions, nprobe=args.nprobe,
        top_k=args.top_k, return_vecs=True,
        max_answer_length=args.max_answer_length, aggregate=args.aggregate, agg_strat=args.agg_strat,
    )
    return outs


def annotate_phrase_vecs(mips, q_ids, questions, answers, titles, phrase_groups, args,
                         pseudo_label_fct=None, is_skips=[]):
    if len(is_skips) == 0:
        is_skips = [False] * len(questions)
        
    assert mips is not None
    batch_size = len(q_ids)
    dummy_group = {
        'doc_idx': -1,
        'start_idx': 0, 'end_idx': 0,
        'answer': '',
        'start_vec': np.zeros(768),
        'end_vec': np.zeros(768),
        'context': '', 'title': [''],
        'doc_uid': ['']
    }

    if args.aggregate: # TODO! check if it's right
        top_k = args.top_k
    else:
        top_k = args.top_k * 2
        
    # Pad phrase groups (two separate top-k coming from start/end, so pad with top_k*2)
    for b_idx, phrase_idx in enumerate(phrase_groups):
        phrase_groups[b_idx] = phrase_groups[b_idx][:top_k]
        
        while len(phrase_groups[b_idx]) < top_k:
            phrase_groups[b_idx].append(dummy_group)
        assert len(phrase_groups[b_idx]) == top_k

    # Flatten phrase groups
    flat_phrase_groups = [phrase for phrase_group in phrase_groups for phrase in phrase_group]
    doc_idxs = [int(phrase_group['doc_idx']) for phrase_group in flat_phrase_groups]
    start_vecs = [phrase_group['start_vec'] for phrase_group in flat_phrase_groups]
    end_vecs = [phrase_group['end_vec'] for phrase_group in flat_phrase_groups]

    # stack vectors
    # [batch_size * topk * 2, d]
    start_vecs = np.stack(start_vecs)
    end_vecs = np.stack(end_vecs)
    zero_mask = np.array([[1] if doc_idx >= 0 else [0] for doc_idx in doc_idxs])
    start_vecs = start_vecs * zero_mask
    end_vecs = end_vecs * zero_mask

    # Reshape
    # [batch_size, topk * 2, d]
    start_vecs = np.reshape(start_vecs, (batch_size, top_k, -1))
    end_vecs = np.reshape(end_vecs, (batch_size, top_k, -1))

    # Dummy targets
    targets = [[None for phrase in phrase_group] for phrase_group in phrase_groups]
    p_targets = [[None for phrase in phrase_group] for phrase_group in phrase_groups]
    
    # Annotate for L_phrase
    if 'phrase' in args.label_strat.split(','):
        # phrase.keys(): dict_keys(['context', 'title', 'doc_idx', 'start_pos', 'end_pos', 'start_idx', 'end_idx', 'score', 'start_vec', 'end_vec', 'answer'])
        targets = [pseudo_label_fct(phrase_group, question, is_skip) for phrase_group, question, is_skip in
                    zip(phrase_groups, questions, is_skips)]
        
        targets = [target[0] for target in targets]
        
        # pass p_targets when it is a soft label
        if targets[0].dtype == 'float32':
            pass
        else:
            targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]
        
    # Annotate for L_doc
    if 'doc' in args.label_strat.split(','):
        raise NotImplementedError
    
    return start_vecs, end_vecs, targets, p_targets

def update_query_vec(
        query_vecs,
        start_vecs=None, end_vecs=None,
        targets=None, is_soft_label=False, 
):
    # Skip if no targets for phrases
    if start_vecs is not None:
        if all([len(t) == 0 for t in targets]):
            return None

    # hotfix for matmul
    query_vecs = query_vecs.float()

    # Compute query embedding
    query_start, query_end = query_vecs.split(query_vecs.shape[-1] // 2, dim=1)

    # Start/end dense logits
    # [batch_size, topk]
    start_logits = query_start.matmul(start_vecs.transpose(1, 2)).sum(axis=1)
    end_logits = query_end.matmul(end_vecs.transpose(1, 2)).sum(axis=1) 
    logits = start_logits + end_logits 

    # L_phrase: MML over phrase-level annotation
    loss = 0.0
    MIN_PROB = 1e-7 
    if is_soft_label:
        kl_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        targets = torch.softmax(torch.stack(targets), dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        start_log_probs = torch.log_softmax(start_logits, dim=-1)
        end_log_probs = torch.log_softmax(end_logits, dim=-1)
        loss = kl_loss_fct(log_probs, targets)
        start_loss = kl_loss_fct(start_log_probs, targets)
        end_loss = kl_loss_fct(end_log_probs, targets)
        
        loss = loss + start_loss + end_loss
            
    elif not all([len(t) == 0 for t in targets]):
        loss = [
            -torch.log(softmax(lg, -1)[tg.long()].sum().clamp(MIN_PROB, 1)) for lg, tg in zip(logits, targets)
            if len(tg) > 0
        ]
        # Start/End only loss
        start_loss = [
            -torch.log(softmax(lg, -1)[tg.long()].sum().clamp(MIN_PROB, 1)) for lg, tg in zip(start_logits, targets)
            if len(tg) > 0
        ]
        end_loss = [
            -torch.log(softmax(lg, -1)[tg.long()].sum().clamp(MIN_PROB, 1)) for lg, tg in zip(end_logits, targets)
            if len(tg) > 0
        ]
        
        loss = torch.stack(loss)
        start_loss = torch.stack(start_loss)
        end_loss = torch.stack(end_loss)
        
        loss = loss + start_loss + end_loss
    return loss

def test_query_vec(args, mips, query_vecs, data, pseudo_label_fct=None):
    is_soft_label = args.pseudo_labeler_type == 'soft'
    assert len(query_vecs) == 1 # Ensure to be instance-level optimzation
    query_vecs = torch.tensor(query_vecs, requires_grad=True, device=device)
    optimizer = SGD([query_vecs], lr=args.learning_rate, momentum=0.99, weight_decay = args.weight_decay)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_train_epochs
    )
    # for element-wise early stop
    skip_indexes = set()
    q_ids, questions, _, titles = data

    # Train arguments
    args.per_device_train_batch_size = int(args.per_device_train_batch_size / args.gradient_accumulation_steps)

    for ep_idx in range(int(args.num_train_epochs)):
        # step1: get top phrases using query vec
        phrase_groups = get_top_phrases_query_vec(mips, questions, query_vecs, args)
            
        # step2: annotate relevance using relevance labeler
        is_skips = np.array([0]*len(questions))
        is_skips[list(skip_indexes)] = 1
        
        svs, evs, tgts, _ = annotate_phrase_vecs(mips, 
                                            q_ids, 
                                            questions, 
                                            None, 
                                            titles, 
                                            phrase_groups, 
                                            args, 
                                            pseudo_label_fct=pseudo_label_fct,
                                            is_skips=is_skips
                                            )  
        
        if args.label_strat == 'doc':
            raise NotImplementedError
        else:
            tgts = [[tgt_ for tgt_ in tgt if tgt_ is not None] for tgt in tgts]
        
        if is_soft_label: # prepare it for early stop
            # top p labeling (Assumption: single instance)
            mml_tgts = torch.Tensor(tgts).squeeze(0)
            mml_tgts = torch.softmax(torch.Tensor(mml_tgts),dim=-1)        
            sorted_idx = torch.argsort(mml_tgts, descending=True)
            pos_cumsum_probs = torch.cumsum(mml_tgts[sorted_idx],dim=-1)
            mml_tgts = torch.zeros_like(mml_tgts).long()
            bools = torch.where(pos_cumsum_probs >= args.pseudo_labeler_p)[0]
            if len(bools) > 0:
                idx = bools[0]
                true_idx = sorted_idx[:idx+1]
                mml_tgts[true_idx] = 1    
            mml_tgts = [i for i, t  in enumerate(mml_tgts) if t]
        else:
            mml_tgts = None

        svs_t = torch.Tensor(svs).to(device)
        evs_t = torch.Tensor(evs).to(device)
        tgts_t = [torch.Tensor(tgt).to(device) for tgt in tgts]
        
        # step3: TouR condition
        # Early-stop if top 1 prediction is positive
        # Otherwise, update query
        if args.top1_earlystop:
            if not is_soft_label and ((0. in tgts_t[0]) or (len(tgts_t[0]) == 0)):
                # logger.info(
                #     f"Ep {ep_idx+1} Early stop -- targets: {tgts_t[0]}"
                # )
                break
            elif is_soft_label and ((0. in mml_tgts) or (len(mml_tgts) == 0)):
                # logger.info(
                #     f"Ep {ep_idx+1} Early stop -- targets: {mml_tgts}"
                # )
                break
            
        # step4: train query vector that doesn't meet prf stop condition
        loss = update_query_vec(
            query_vecs=query_vecs,
            start_vecs=svs_t,
            end_vecs=evs_t,
            targets=tgts_t,
            is_soft_label=is_soft_label
        )
        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(torch.ones_like(scaled_loss))
        else:
            # element-wise backward
            # https://discuss.pytorch.org/t/error-grad-can-be-implicitly-created-only-for-scalar-outputs/38102/18
            loss.backward(torch.ones_like(loss))

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(query_vecs, args.max_grad_norm)
        
        # if is_soft_label:
        #     logger.info(
        #         f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f} targets: {mml_tgts} LR: {scheduler.get_last_lr()}"
        #     )
        # else:
        #     logger.info(
        #         f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f} targets: {tgts_t[0]} LR: {scheduler.get_last_lr()}"
        #     )
        
        optimizer.step()
        scheduler.step()
                
    query_vecs = query_vecs.cpu().detach().numpy()
            
    return query_vecs

def do_tour(args, mips, q_ids=None, questions=None, answers=None, titles=None, query_vecs=None):
    if args.eval_psg:
        raise NotImplementedError

    task = 'odqa'
    batch_size = args.per_device_train_batch_size

    # load a cross-encoder    
    ce_model = CrossEncoder(
        model_name_or_path=args.pseudo_labeler_name_or_path,
        pseudo_labeler_type=args.pseudo_labeler_type,
        pseudo_labeler_p=args.pseudo_labeler_p,
        pseudo_labeler_temp=args.pseudo_labeler_temp,
        use_cuda=args.cuda,
        task=task
    )
    pseudo_label_fct = ce_model.do_labeling

    # update query vector
    logger.info(f"Start TouR on {len(questions)} questions")
    updated_query_vecs = np.copy(query_vecs)
    for i in tqdm(range(0, len(q_ids), batch_size)):
        updated_query_vecs_batch = test_query_vec(
            args, mips, query_vecs[i:i+batch_size],
            data=[q_ids[i:i+batch_size], questions[i:i+batch_size], answers[i:i+batch_size], titles[i:i+batch_size]],
            pseudo_label_fct=pseudo_label_fct
        )
        update_batch_size = len(updated_query_vecs_batch)
        updated_query_vecs[i:i+update_batch_size] = updated_query_vecs_batch
    query_vecs = updated_query_vecs
    logger.info(f"Finish TouR")
    
    # evaluate query vectors after TouR
    new_args = copy.deepcopy(args)
    new_args.save_pred = True
    new_args.aggregate = True
    em_top1, _, em_topk, _ = evaluate(new_args, mips, query_vec=query_vecs)

    # aggregate scores from dual-encoder and cross-encoder
    # load predictions from dual-encoder
    total = len(questions)
    pred_dir = os.path.join(args.load_dir, 'pred')
    pred_path = os.path.join(
        pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}_top{args.top_k}.pred'
    )
    qas_to_rerank = load_json(pred_path)

    # aggregate cross-encoder scores
    predictions = ce_model.do_rerank(
        qas_to_rerank,
        rerank_k=args.top_k,
        rerank_lambda=args.rerank_lambda,
    )
        
    with open(args.test_path) as f:
        answers = {d['id']:d['answers'] for d in json.load(f)['data']}

    # evaluate predictions after aggregation
    em_top1 = evaluate_prediction_rerank(predictions, answers, args)

    # save predictions
    pred_path = args.test_path.split("/")[-1].replace(".json","_tour.pred")
    pred_path = os.path.join(args.output_dir, pred_path)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)

    logger.info(f"Finish TouR")
    logger.info(f"Acc={em_top1:.2f} | Acc@{new_args.top_k}={em_topk:.2f}")
    logger.info(f"Predictions are saved in {pred_path}")

def evaluate_prediction_rerank(predictions, answers, args):
    # Get em/f1
    ems = []
    for (_, prediction) in predictions.items():
        groundtruth = answers[prediction['q_id']]
        if len(groundtruth)==0:
            ems.append(0)
            continue
        top1_pred = prediction['prediction'][0]
        match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
        ems.append(max([match_fn(top1_pred, gt) for gt in groundtruth]))
    final_em = np.mean(ems) * 100
    return final_em

def load_json(fi):
    logging.info(f'Loading {fi}')

    results = []
    with open(fi) as f:
        data = json.load(f)
        for i, (q_id, value) in enumerate(data.items()):
            if 'after_prf' in value:
                value = value['after_prf']
            item = {
                'q_id': q_id
            }
            item.update(value)
            results.append(item)
            logging.info(f'Loaded {i + 1} Items from {fi}') if i % 1000 == 0 else None
    # logging.info(f'Loaded {i + 1} Items from {fi}')
    return results

def evaluate(args, mips, query_vec):
    # load dataset and encode queries
    qids, questions, answers, _ = load_qa_pairs(args.test_path, args)

    # search
    step = args.eval_batch_size
    logger.info(f'Aggergation strategy used: {args.agg_strat}')
    predictions = []
    evidences = []
    titles = []
    scores = []
    se_poss = []
    for q_idx in tqdm(range(0, len(questions), step)):
        result = mips.search(
            query_vec[q_idx:q_idx+step],
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.top_k, max_answer_length=args.max_answer_length,
            aggregate=args.aggregate, agg_strat=args.agg_strat, return_sent=args.return_sent
        )
        prediction = [[ret['answer'] for ret in out][:args.top_k] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out][:args.top_k] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out][:args.top_k] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out][:args.top_k] if len(out) > 0 else [-1e10] for out in result]
        se_pos = [[(ret['start_pos'], ret['end_pos']) for ret in out][:args.top_k] if len(out) > 0 else [(0,0)] for out in result]
        predictions += prediction
        evidences += evidence
        titles += title
        scores += score
        se_poss += se_pos

    eval_fn = evaluate_results
    return eval_fn(predictions, qids, questions, answers, args, evidences, scores, titles, se_positions=se_poss)

if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    options.add_qsft_options()
    
    options.parser.add_argument("--pseudo_labeler_name_or_path", help='a path for the pseudo labeler')
    options.parser.add_argument("--pseudo_labeler_type", default='hard', choices=['hard','soft'])
    options.parser.add_argument("--pseudo_labeler_p", type=float, default=0.5)
    options.parser.add_argument("--pseudo_labeler_temp", type=float, default=1.0)
    options.parser.add_argument("--top1_earlystop", action='store_true')
    options.parser.add_argument("--rerank_lambda", type=float, default=0.1)
    args = options.parse()

    assert args.pseudo_labeler_temp > 0

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.getLogger("eval_phrase_retrieval").setLevel(logging.DEBUG)
    logging.getLogger("densephrases.utils.open_utils").setLevel(logging.DEBUG)

    # load the phrase index
    mips = load_phrase_index(args)

    # load the query encoder
    device = 'cuda' if args.cuda else 'cpu'
    logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
    logger.info("Args: {}".format(args))
    query_encoder, tokenizer, _ = load_encoder(device, args, query_only=True)

    # load test questions
    q_ids, questions, answers, titles = load_qa_pairs(
        args.test_path, 
        args, 
        shuffle=False
    )

    # embed queries
    query_vecs = embed_all_query(questions, args, query_encoder, tokenizer)
    
    # do TouR
    do_tour(args, mips, q_ids, questions, answers, titles, query_vecs)