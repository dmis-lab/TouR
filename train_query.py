from builtins import NotImplementedError
import torch
import random
import numpy as np
import logging
import copy

from tqdm import tqdm
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.eval_utils import (
    drqa_exact_match_score, drqa_regex_match_score,
    drqa_metric_max_over_ground_truths
)
from eval_phrase_retrieval import evaluate
from densephrases import Options
from cross_encoder import CrossEncoder
from eval_phrase_retrieval import embed_all_query
from torch.nn.functional import softmax
from numpy.random import default_rng
rng = default_rng()

from transformers import get_linear_schedule_with_warmup

from torch.optim import SGD

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def get_top_phrases(mips, q_ids, questions, answers, titles, query_encoder, tokenizer, batch_size, args):
    # Search
    step = batch_size
    multivec_len = args.multivec_len
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )
    for q_idx in tqdm(range(0, len(questions), step)):
        outs = list(query2vec(questions[q_idx:q_idx + step]))
        # [(len(outs), multivec_len, d)]
        start = np.concatenate([out[0] for out in outs], 0).reshape(len(outs), multivec_len, -1)
        # [(len(outs), multivec_len, d)] 
        end = np.concatenate([out[1] for out in outs], 0).reshape(len(outs), multivec_len, -1)      
        # [(len(outs), multivec_len, 2 * d)]
        query_vec = np.concatenate([start, end], 2)                                                  

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx + step], nprobe=args.nprobe,
            top_k=args.top_k, return_vecs=True,
            max_answer_length=args.max_answer_length, aggregate=args.aggregate, agg_strat=args.agg_strat,
        )
        yield (
            q_ids[q_idx:q_idx + step], questions[q_idx:q_idx + step], answers[q_idx:q_idx + step],
            titles[q_idx:q_idx + step], outs
        )

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


def annotate_phrase_vecs(mips, q_ids, questions, answers, titles, phrase_groups, args, label_strategy='gold',
                         pseudo_label_fct=None, is_skips=[]):
    if len(is_skips) == 0:
        is_skips = [False] * len(questions)
    is_psg = args.eval_psg
        
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
        match_fns = [
            drqa_regex_match_score if args.regex or ('trec' in q_id.lower()) else drqa_exact_match_score for q_id in
            q_ids
        ]
        if label_strategy == 'gold':
            targets = [
                [drqa_metric_max_over_ground_truths(match_fn, phrase['answer'], answer_set) for phrase in phrase_group]
                for phrase_group, answer_set, match_fn in zip(phrase_groups, answers, match_fns)
            ]
        elif label_strategy == 'pseudo':
            # phrase.keys(): dict_keys(['context', 'title', 'doc_idx', 'start_pos', 'end_pos', 'start_idx', 'end_idx', 'score', 'start_vec', 'end_vec', 'answer'])
            targets = [pseudo_label_fct(phrase_group, question, is_skip, is_psg=is_psg) for phrase_group, question, is_skip in
                       zip(phrase_groups, questions, is_skips)]
            
            targets = [target[0] for target in targets]
        elif label_strategy.startswith('top'):
            K = len(phrase_groups[0])
            k = int(label_strategy.replace("top",""))
            target = np.zeros(K)
            target[:k] = 1
            targets = [target for phrase_group in phrase_groups]
        else:
            raise NotImplementedError('invalid strategy')
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
        student_temp=1.0
):

    # Skip if no targets for phrases
    if start_vecs is not None:
        if all([len(t) == 0 for t in targets]):
            return None

    # hotfix for matmul
    query_vecs = query_vecs.float()

    # Compute query embedding
    query_start, query_end = query_vecs.split(query_vecs.shape[-1] // 2, dim=2)

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
        log_probs = torch.log_softmax(logits/student_temp, dim=-1)
        start_log_probs = torch.log_softmax(start_logits/student_temp, dim=-1)
        end_log_probs = torch.log_softmax(end_logits/student_temp, dim=-1)
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
    args.per_device_eval_batch_size = int(args.per_device_eval_batch_size / args.gradient_accumulation_steps)

    for ep_idx in range(int(args.num_train_epochs)):
        # step1: get top phrases using query vec
        phrase_groups = get_top_phrases_query_vec(mips, questions, query_vecs, args)
            
        # step2: annotate relevance using relevance labeler
        is_skips = np.array([0]*len(questions))
        is_skips[list(skip_indexes)] = 1
        label_strategy = args.label_strategy if args.label_strategy.startswith('top') else 'pseudo'
        
        svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, 
                                            q_ids, 
                                            questions, 
                                            None, 
                                            titles, 
                                            phrase_groups, 
                                            args, 
                                            label_strategy='none' if pseudo_label_fct is None else label_strategy, 
                                            pseudo_label_fct=pseudo_label_fct,
                                            is_skips=is_skips
                                            )  
        
        if args.label_strat == 'doc':
            raise NotImplementedError
        else:
            tgts = [[tgt_ for tgt_ in tgt if tgt_ is not None] for tgt in tgts]
        
        if args.eval_psg:
            raise NotImplementedError
        else:
            keys = [f"{pg['title']};{pg['start_pos']};{pg['end_pos']}" for pg in phrase_groups[0]]
        

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
        
        # step3: TQR condition
        # Early-stop if top 1 prediction is positive
        # Otherwise, update query
        if args.top1_earlystop:
            if not is_soft_label and ((0. in tgts_t[0]) or (len(tgts_t[0]) == 0)):
                logger.info(
                    f"Ep {ep_idx+1} Early stop -- targets: {tgts_t[0]}"
                )
                break
            elif is_soft_label and ((0. in mml_tgts) or (len(mml_tgts) == 0)):
                logger.info(
                    f"Ep {ep_idx+1} Early stop -- targets: {mml_tgts}"
                )
                break
            
        # step4: train query vector that doesn't meet prf stop condition
        loss = update_query_vec(
            query_vecs=query_vecs,
            start_vecs=svs_t,
            end_vecs=evs_t,
            targets=tgts_t,
            is_soft_label=is_soft_label,
            student_temp=args.student_temp
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
        
        if is_soft_label:
            logger.info(
                f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f} targets: {mml_tgts} LR: {scheduler.get_last_lr()}"
            )
        else:
            logger.info(
                f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f} targets: {tgts_t[0]} LR: {scheduler.get_last_lr()}"
            )
        
        optimizer.step()
        scheduler.step()
                
    query_vecs = query_vecs.cpu().detach().numpy()
            
    return query_vecs

def do_test_query(args, mips, query_encoder=None, tokenizer=None, q_ids=None, questions=None, answers=None, titles=None, query_vecs=None):
    if args.eval_psg:
        raise NotImplementedError
    else:
        task = 'odqa'
        
    ce_model = CrossEncoder(
        model_name_or_path=args.pseudo_labeler_name_or_path,
        pseudo_labeler_type=args.pseudo_labeler_type,
        pseudo_labeler_p=args.pseudo_labeler_p,
        pseudo_labeler_temp=args.pseudo_labeler_temp,
        use_cuda=args.cuda,
        no_title=args.no_title, 
        title_delimiter=args.title_delimiter,
        input_type=args.input_type,
        task=task,
        cache_path=args.ce_cache_path,
        overwrite_cache=args.overwrite_cache
    )
    pseudo_label_fct = ce_model.do_labeling

    batch_size = args.per_device_eval_batch_size

    logger.info(f"Before applying TouR on {len(questions)} questions")
    if args.no_eval_before_tqr:
        pass
    elif args.eval_psg:
        raise NotImplementedError
    elif not args.is_kilt:
        new_args = copy.deepcopy(args)
        new_args.top_k = args.pred_top_k
        new_args.save_pred = False
        new_args.aggregate = True
        result = evaluate(new_args, mips, query_encoder=query_encoder,
                                            tokenizer=tokenizer, query_vec=query_vecs)

        em_top1 = result['em_top1']
        em_topk = result['em_topk']
    else:
        raise NotImplementedError
    # update query vector
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

    logger.info(f"After apply TouR on {len(questions)} questions")
    if args.eval_psg:
        raise NotImplementedError
    elif not args.is_kilt:
        new_args = copy.deepcopy(args)
        new_args.top_k = args.pred_top_k
        new_args.save_pred = False
        new_args.aggregate = True
        result = evaluate(new_args, mips, query_encoder=query_encoder,
                                tokenizer=tokenizer, query_vec=query_vecs)


        em_top1 = result['em_top1']
        f1_top1 = result['f1_top1']
        em_topk = result['em_topk']
        f1_topk = result['f1_topk']
        logger.info(f"Acc={em_top1:.2f} | F1={f1_top1:.2f}")
        logger.info(f"Acc@{new_args.top_k}={em_topk:.2f} | F1@{new_args.top_k}={f1_topk:.2f}")
    else:
        raise NotImplementedError
    
if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    options.add_qsft_options()
    
    options.parser.add_argument("--label_strategy", default="gold")
    options.parser.add_argument("--pseudo_labeler_name_or_path")
    options.parser.add_argument("--pseudo_labeler_type", default='top_p_hard', choices=['top_p_hard','soft'])
    options.parser.add_argument("--pseudo_labeler_p", type=float, default=0.5)
    options.parser.add_argument("--pseudo_labeler_temp", type=float, default=1.0)
    options.parser.add_argument("--student_temp", type=float, default=1.0)
    options.parser.add_argument("--pred_top_k", type=int, default=10)
    options.parser.add_argument("--multivec_len", type=int, default=1)
    options.parser.add_argument("--top1_earlystop", action='store_true')
    options.parser.add_argument("--is_entity_question", action='store_true')
    options.parser.add_argument("--no_eval_before_tqr", action='store_true')
    options.parser.add_argument("--no_title", action='store_true')
    options.parser.add_argument("--save_psg_pred", action='store_true')
    options.parser.add_argument("--optimizer", default='adam', choices=['adam','sgd', 'sgdm0.9', 'sgdm0.99','rmsprop'])
    options.parser.add_argument("--input_type", default='3sent', choices=['3sent', '1sent', '5sent', 'whole'])
    options.parser.add_argument("--title_delimiter", default='sep', choices=['sep','space'])
    options.parser.add_argument("--no_lr_schedule", default=0, type=int)
    options.parser.add_argument("--grid_option", default=1, type=int)
    options.parser.add_argument("--ce_cache_path", default=None)
    options.parser.add_argument("--overwrite_cache", action='store_true')
    options.parser.add_argument("--online_learning", action='store_true')
    options.parser.add_argument("--keep_prev_topk", action='store_true')
    options.parser.add_argument("--keep_cur_topk", action='store_true')
    options.parser.add_argument("--keep_prev_topk_tgts", type=str, default='top_p')
    options.parser.add_argument("--noise_type", type=str, default='none')
    options.parser.add_argument("--noise_norm", type=float, default=0.0)
    options.parser.add_argument("--update_mode", default='gradient', choices=['gradient', 'interpolation_pos'])
    options.parser.add_argument("--interpolation_beta", type=float, default=1.0)
    options.parser.add_argument("--interpolation_k", type=int, default=10)
    args = options.parse()

    assert args.pseudo_labeler_temp > 0
    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    elif args.run_mode == 'test_query_vec':
        logging.getLogger("eval_phrase_retrieval").setLevel(logging.DEBUG)
        logging.getLogger("densephrases.utils.open_utils").setLevel(logging.DEBUG)

        # Train
        mips = load_phrase_index(args)

        # Query encoder
        device = 'cuda' if args.cuda else 'cpu'
        logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
        logger.info("Args: {}".format(args))
        query_encoder, tokenizer, _ = load_encoder(device, args, query_only=True)

        # Load test questions
        q_ids, questions, answers, titles = load_qa_pairs(
            args.test_path, 
            args, 
            shuffle=False,
            draft_num_examples=args.draft_num_examples
        )

        if len(q_ids) == 2:
            q_ids = q_ids[0]

        query_vecs = embed_all_query(questions, args, query_encoder, tokenizer)
        
        do_test_query(args, mips, query_encoder, tokenizer, q_ids, questions, answers, titles, query_vecs)