#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch
import logging
import time
import json
from spacy.lang.en import English
from itertools import chain
import numpy as np
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModel,
    PreTrainedModel,
    AutoModelForSequenceClassification,
)
import os
import copy
from tqdm import tqdm

try:
    import apex
    from apex import amp
    apex.amp.register_half_function(torch, "einsum")
    _has_apex = True
except ImportError:
    _has_apex = False

# From R2-D2
class BaselineRerankerQueryBuilder(object):

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_context_token_id = self.tokenizer.convert_tokens_to_ids("madeupword0000")
        self.start_title_token_id = self.tokenizer.convert_tokens_to_ids("madeupword0001")

    def tokenize_and_convert_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @property
    def num_special_tokens_to_add(self):
        return self.tokenizer.num_special_tokens_to_add(pair=True)

    def __call__(self, question, passages, numerized=False):
        if not numerized:
            question = self.tokenize_and_convert_to_ids(question)
            passages = [(self.tokenize_and_convert_to_ids(item[0]), self.tokenize_and_convert_to_ids(item[1])) for item in passages]

        cls = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])
        sep = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
        eos = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])

        input_ids_list = []

        for passage in passages:
            input_ids = cls + question + sep + sep
            input_ids.extend([self.start_title_token_id] + passage[0])
            input_ids.extend([self.start_context_token_id] + passage[1] + eos)

            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length-1] + eos

            input_ids_list.append(input_ids)
    
        seq_len = max(map(len, input_ids_list))

        input_ids_tensor = torch.ones(len(input_ids_list), seq_len).long()
        attention_mask_tensor = torch.zeros(len(input_ids_list), seq_len).long()

        for batch_index, input_ids in enumerate(input_ids_list):

            for sequence_index, value in enumerate(input_ids):
                input_ids_tensor[batch_index][sequence_index] = value
                attention_mask_tensor[batch_index][sequence_index] = 1.

        features = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor
        }

        return features

class BaselineReranker(torch.nn.Module):
    """ Baseline passage reranker used in the paper. """

    def __init__(self, config, encoder):
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.vt = torch.nn.Linear(config.hidden_size, 1, bias=False)

        self.init_weights(type(self.encoder))

    def init_weights(self, clz):
        """ Applies model's weight initialization to all non-pretrained parameters of this model"""
        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.encoder, module))

    def forward(self, batch):
        """
        The input looks like:
        [CLS] Q [SEP] <t> title <c> context [EOS]
        """

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }

        outputs = self.encoder(**inputs)[1]

        scores = self.vt(outputs)
        scores = scores.view(1,-1)

        return scores
        
logger = logging.getLogger(__name__)
CUDA = torch.cuda.is_available()

sentencizer = English()
sentencizer.add_pipe('sentencizer')

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
    return results

def dump_json(items, fi):
    logging.info(f'Dumping {len(items)} items into {fi}')
    with open(fi, 'w') as f:
        json.dump(items, f)

def to_fp16(model):
    if _has_apex:
        model = amp.initialize(model, opt_level="O1")
    else:
        model = model.half()
    return model


def sort_based_on_index(l, index):
    # input
    #   l: ['Linda Davis', 'Reba McEntire and Linda Davis', 'Reba McEntire', 'Reba McEntire and Linda Davis', 'Linda Davis', 'Linda Davis', 'Linda Davis', 'Linda Kaye Davis', 'LeAnn Rimes', 'LeAnn Rimes', 'Barbra Jean.', 'Odia Coates', 'Gloria Loring', 'Reba McEntire', 'Dolly Parton', 'Reba McEntire', 'Millie Jackson', 'Vonda Shepard', 'Cheyenne', 'Linda Davis', 'Laura Manuel', 'Barbra Jean', 'Chris Stapleton.', 'Gloria Loring', 'Brett Beavers', 'Katy Perry', 'Barbra Jean', 'Barbra Jean', 'Jenna Ushkowitz;', 'Wretch 32.', 'Eva Simons', 'Jacob Banks', 'Lucie Silvas', 'Anne Murray', 'Demi Lovato', 'Barbra Jean', 'Cheyenne', 'Judith Glory Hill', 'Kashif,', 'Holly Dunn', 'Donna Summer', 'Charly McClain,', 'Regina Love', 'Elizabeth', 'Trisha Yearwood', 'Barbra Jean', 'Julie Doiron,', 'Lauren Daigle', 'Troy Seals', 'Barbra Jean']
    #   index: [6, 0, 4, 26, 8, 18, 45, 13, 27, 10, 36, 2, 1, 35, 15, 49, 19, 43, 5, 7, 9, 41, 21, 14, 42, 3, 37, 16, 47, 17, 23, 11, 20, 48, 12, 22, 44, 25, 34, 30, 28, 24, 39, 29, 46, 31, 33, 38, 32, 40]
    # output
    #   ['Linda Davis', 'Linda Davis', 'Linda Davis', 'Barbra Jean', 'LeAnn Rimes', 'Cheyenne', 'Barbra Jean', 'Reba McEntire', 'Barbra Jean', 'Barbra Jean.', 'Cheyenne', 'Reba McEntire', 'Reba McEntire and Linda Davis', 'Barbra Jean', 'Reba McEntire', 'Barbra Jean', 'Linda Davis', 'Elizabeth', 'Linda Davis', 'Linda Kaye Davis', 'LeAnn Rimes', 'Charly McClain,', 'Barbra Jean', 'Dolly Parton', 'Regina Love', 'Reba McEntire and Linda Davis', 'Judith Glory Hill', 'Millie Jackson', 'Lauren Daigle', 'Vonda Shepard', 'Gloria Loring', 'Odia Coates', 'Laura Manuel', 'Troy Seals', 'Gloria Loring', 'Chris Stapleton.', 'Trisha Yearwood', 'Katy Perry', 'Demi Lovato', 'Eva Simons', 'Jenna Ushkowitz;', 'Brett Beavers', 'Holly Dunn', 'Wretch 32.', 'Julie Doiron,', 'Jacob Banks', 'Anne Murray', 'Kashif,', 'Lucie Silvas', 'Donna Summer']
    new_list = []
    for ind in index:
        new_list.append(l[ind])
    return new_list

def get_output_format(qas, prediction_indices, output_scores):
    assert len(qas) == len(prediction_indices)
    output = {}
    for sample, scores, prediction_index in zip(qas, output_scores, prediction_indices):
        top_k = len(prediction_index)
        
        q_id = sample['q_id']
        sample['title'] = sort_based_on_index(sample['title'][:top_k], prediction_index) + sample['title'][top_k:]
        sample['prediction'] = sort_based_on_index(sample['prediction'][:top_k], prediction_index) + sample['prediction'][top_k:]
        sample['score'] = sort_based_on_index(scores[:top_k], prediction_index) + len(scores[top_k:]) * [-1]
        sample['evidence'] = sort_based_on_index(sample['evidence'][:top_k], prediction_index) + sample['evidence'][top_k:]
        sample['se_pos'] = sort_based_on_index(sample['se_pos'][:top_k], prediction_index) + sample['se_pos'][top_k:]
        output[q_id] = sample
    
    return output

def tag_phrase_in_passage(se_pos, phrase, passage):
    s_pos, e_pos = se_pos
    passage = passage[:s_pos] + ' [S] ' + phrase + ' [E] ' + passage[e_pos:]
    is_valid = True
    
    return passage, is_valid

def tag_phrase_in_sentence(se_pos, phrase, doc, num_sent=3):
    s_pos, e_pos = se_pos
    new_s_pos, new_e_pos = -1 , -1
    s_sent_pos, e_sent_pos = -1, -1
    sent_chars = [(sent.start_char, sent.end_char) for sent in doc.sents]

    for i, (start_char, end_char) in enumerate(sent_chars):
        # start_char = sent.start_char
        # end_char = sent.end_char
        prev_start_char = 0
        next_end_char = len(doc.text)
        
        # TODO. need refactoring
        if num_sent == 1: # target sentence that has a phrase
            prev_start_char = sent_chars[i][0]
            next_end_char = sent_chars[i][1]
        elif num_sent == 3: # one sentence on the left and right of the target sentence
            if i > 0:
                prev_start_char = sent_chars[i-1][0]
            else:
                prev_start_char = sent_chars[i][0]
            
            if i < len(sent_chars) - 1:
                next_end_char = sent_chars[i+1][1]
            else:
                next_end_char = sent_chars[i][1]
        elif num_sent == 5: # two sentences on the left and right of the target sentence
            if i > 1:
                prev_start_char = sent_chars[i-2][0]
            elif i > 0:
                prev_start_char = sent_chars[i-1][0]
            else:
                prev_start_char = sent_chars[i][0]
            
            if i < len(sent_chars) - 2:
                next_end_char = sent_chars[i+2][1]
            elif i < len(sent_chars) - 1:
                next_end_char = sent_chars[i+1][1]
            else:
                next_end_char = sent_chars[i][1]

        if s_pos >= start_char and s_pos<end_char:
            s_sent_pos = prev_start_char
            new_s_pos = s_pos - prev_start_char
            new_e_pos = e_pos - prev_start_char
        
        if e_pos > start_char and e_pos<=end_char:
            e_sent_pos = next_end_char

    is_valid = False
    try:
        assert (s_sent_pos <= e_sent_pos)
        assert (new_s_pos <= new_e_pos)

        assert ((s_sent_pos >= 0) and (e_sent_pos >= 0))
        assert ((new_s_pos >= 0) and (new_e_pos >= 0))

        new_sent = doc.text[s_sent_pos:e_sent_pos]
        assert new_sent[new_s_pos:new_e_pos] == phrase
        new_pred = '[S] ' + phrase + ' [E]'
        new_sent = new_sent[:new_s_pos]  + new_pred + new_sent[new_e_pos:]
        
        is_valid = True
        return new_sent, is_valid
    except:
        return "", is_valid

class CE_Cache(object):
    def __init__(self):
        self.cache = {}
        
    def get_cache_key(self, question, evidence, s_pos=None, e_pos=None):
        cache_key = ';'.join([question, evidence, str(s_pos), str(e_pos)])
        return cache_key
    
    def is_cached(self, question, evidence, s_pos=None, e_pos=None):
        cache_key = self.get_cache_key(question, evidence, s_pos, e_pos)
        return cache_key in self.cache
    
    def set_cache(self, question, evidence, s_pos=None, e_pos=None, logit=None):
        cache_key = self.get_cache_key(question, evidence, s_pos, e_pos)
        self.cache[cache_key] = logit
        
    def get_cache(self, question, evidence, s_pos=None, e_pos=None):
        cache_key = self.get_cache_key(question, evidence, s_pos, e_pos)
        return self.cache[cache_key]
    
class CrossEncoder(object):
    def __init__(self, 
                 model_name_or_path='',
                 no_title=False, 
                 title_delimiter='sep', 
                 task='odqa', 
                 verbose=False, 
                 input_type='3sent', 
                 use_cuda=True,
                 pseudo_labeler_type='',
                 pseudo_labeler_k='',
                 pseudo_labeler_p='',
                 pseudo_labeler_temp='',
                 minimum_pos=0,
                 ):
        self.no_title = no_title
        self.title_delimiter = title_delimiter
        self.task = task
        self.been = 0 # for logging
        self.verbose = verbose
        self.input_type = input_type
        
        if input_type == '1sent':
            self.num_sent = 1
        elif input_type == '3sent':
            self.num_sent = 3
        elif input_type == '5sent':
            self.num_sent = 5
        elif input_type == 'whole':
            self.num_sent = -1
        else:
            raise NotImplementedError  
        
        if use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.model, self.tokenizer = self.load_model(model_name_or_path)
        
        self.max_seq_length = 512
        self.no_title = no_title
        self.been = 0
        self.ce_cache = CE_Cache()
            
        self.pseudo_labeler_type = pseudo_labeler_type
        self.pseudo_labeler_k = pseudo_labeler_k
        self.pseudo_labeler_p = pseudo_labeler_p
        self.pseudo_labeler_temp = pseudo_labeler_temp
        self.minimum_pos = minimum_pos
        
    def load_model(self, model_name_or_path):
        logger.info(f'[{self.task}] Loading model from: {model_name_or_path}')
        
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        ).to(self.device)
        model = model.eval()
        
        return model, tokenizer

    # TODO! need to merge it with do_labeler
    def tokenize_for_reranker(self, batch_qas, cuda):
        num_pred = len(batch_qas[0]['prediction'])
        questions = [([qas['question']] * len(qas['evidence']))[:num_pred] for qas in batch_qas]
        evidences = [qas['evidence'][:num_pred] for qas in batch_qas]
        titles = [qas['title'][:num_pred] for qas in batch_qas]
        se_poses = [qas['se_pos'][:num_pred] for qas in batch_qas]
        predictions = [qas['prediction'][:num_pred] for qas in batch_qas]
        
        # Flatten out
        ft_questions = list(chain(*questions))   
        ft_evidences = list(chain(*evidences))
        ft_titles = list(chain(*titles))    
        ft_se_poses = list(chain(*se_poses))
        ft_predictions = list(chain(*predictions))

        if 'sent' in self.input_type: 
            # Preprocess evidences (3 sents , tag phrase)
            ft_evidences = list(sentencizer.pipe(ft_evidences))  
            ft_evidences = [tag_phrase_in_sentence(se_pos, pred, ev, self.num_sent)[0] for ev, se_pos, pred in zip(ft_evidences, ft_se_poses, ft_predictions)]
        elif self.input_type == 'whole':
            ft_evidences = [tag_phrase_in_passage(se_pos, pred, ev)[0] for ev, se_pos, pred in zip(ft_evidences, ft_se_poses, ft_predictions)]
        else:
            raise NotImplementedError
        
        # hotfix
        ft_titles = [[t] if isinstance(t,str) else t for t in ft_titles][:]
        
        assert len(ft_titles[0]) == 1
            
        if not self.no_title:
            if self.title_delimiter == 'sep':
                sep_token = self.tokenizer.sep_token
            elif self.title_delimiter == 'space':
                sep_token = ' '
            else:
                raise NotImplementedError
            ft_evidences = [f"{title[0]} {sep_token} {ev}" for ev, title in zip(ft_evidences, ft_titles)]
        
        if self.been < 3:
            print(ft_questions[0])
            print(ft_evidences[0])
            self.been += 1
        
        # Tokenize
        inputs = self.tokenizer(
            ft_questions,
            ft_evidences,
            truncation=True,
            max_length=512,
            return_tensors='pt', 
            padding="longest"
        )
        
        # Un-flatten
        inputs = {k: v.reshape(len(batch_qas), v.shape[0]//len(batch_qas), -1) for k,v in inputs.items()}
        output = {k: v.cuda() for k, v in inputs.items()} if cuda else inputs
        return output

    def hard_label(self, outputs, top_p):
        pos_logits = outputs[:,1]/self.pseudo_labeler_temp
        sorted_idx = torch.argsort(pos_logits, descending=True)
        pos_probs = torch.softmax(pos_logits, dim=-1)
        pos_cumsum_probs = torch.cumsum(pos_probs[sorted_idx],dim=-1)
        all_predictions = torch.zeros_like(pos_logits).long()
        bools = torch.where(pos_cumsum_probs >= top_p)[0]
        if len(bools) > 0:
            idx = bools[0]
            true_idx = sorted_idx[:idx+1]
            all_predictions[true_idx] = 1
        
        all_predictions = all_predictions.cpu().detach().numpy()
        return all_predictions
    
    def soft_label(self, outputs):
        # pos_probs = torch.sigmoid(outputs/self.pseudo_labeler_temp) # Note! this is sigmoid
        pos_probs = outputs[:,1]/self.pseudo_labeler_temp # Note! this is sigmoid
        pos_probs = pos_probs.cpu().detach().numpy()
        return pos_probs
    
    def do_labeling(self, phrase_group, question, is_skip):
        if is_skip:
            labels = np.array([0] * len(phrase_group))
            return -1, labels
        
        # phrase_group: a list of "phrase"s.
        # phrase.keys(): dict_keys(['context', 'title', 'doc_idx', 'start_pos', 'end_pos', 'start_idx', 'end_idx', 'score', 'start_vec', 'end_vec', 'answer'])
        
        # filter new_phrase_groups
        new_phrase_group = []
        for _, phrase in enumerate(phrase_group):
            has_cache = self.ce_cache.is_cached(question, phrase['context'], phrase['start_pos'], phrase['end_pos'])
            
            if not has_cache:
                new_phrase_group.append(phrase)
        
        # inference using new_phrase_groups
        if len(new_phrase_group) > 0:
            with torch.inference_mode():
                n_phrases = len(new_phrase_group)
                docs = list(sentencizer.pipe([phrase['context'] for phrase in new_phrase_group]))
                evidences_and_is_valids = [tag_phrase_in_sentence([phrase['start_pos'],phrase['end_pos']], phrase['answer'], doc, num_sent=self.num_sent) for (phrase, doc) in zip(new_phrase_group, docs)]
                evidences = [s[0] for s in evidences_and_is_valids]
                if not self.no_title:
                    assert len(new_phrase_group[0]['title']) == 1
                    evidences = [f'{phrase["title"][0]} {self.tokenizer.sep_token} {ev}' for ev, phrase in zip(evidences, new_phrase_group)]

                # logging
                if self.been < 2:
                    print(question)
                    print(evidences[0])
                    self.been += 1
                
                sentence_pairs =(
                    ([question]*n_phrases, evidences)
                )
                encodings = self.tokenizer(
                    *sentence_pairs, 
                    padding='longest', 
                    max_length=self.max_seq_length, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encodings = encodings.to(self.device)
                logits = self.model(**encodings)[0]
                
                # caching
                for phrase, logit in zip(new_phrase_group, logits):
                    logit = logit.detach().cpu().tolist()
                    self.ce_cache.set_cache(question, phrase['context'], phrase['start_pos'], phrase['end_pos'], logit)
        
        # load probs from cache
        logits = []
        for _, phrase in enumerate(phrase_group):
            logit = self.ce_cache.get_cache(question, phrase['context'], phrase['start_pos'], phrase['end_pos'])
            logits.append(logit)
        
        logits = torch.Tensor(logits)
        scores = None
        if self.pseudo_labeler_type == 'hard':
            all_predictions = self.hard_label(logits, self.pseudo_labeler_p)
        elif self.pseudo_labeler_type == 'soft':
            all_predictions = self.soft_label(logits)
        else:
            raise NotImplementedError
        
        scores = logits[:,1]
        
        return all_predictions, scores
    
    def do_rerank(self, qas, use_cuda=True, bsz=1, rerank_lambda=0.1, rerank_k=10):
        def forward(inputs):
            batch_size, top_k, seq_length = inputs['input_ids'].shape
            inputs['input_ids'] = inputs['input_ids'].reshape(-1, seq_length).to(self.device)
            if 'token_type_ids' in inputs:
                inputs['token_type_ids'] = inputs['token_type_ids'].reshape(-1, seq_length).to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].reshape(-1, seq_length).to(self.device)
            
            logits = self.model(**inputs)[0]
            logits = logits.reshape(batch_size, top_k, -1)
            return logits.detach().cpu().tolist()

        outputs = []
        output_scores = []
        
        logger.info(f'Embedding {len(qas)} inputs in {len(list(range(0, len(qas), bsz)))} batches, rerank_k{rerank_k}:')
        with torch.inference_mode():
            for j, batch_start in tqdm(enumerate(range(0, len(qas), bsz)), total=int(len(qas)/bsz)):
                batch = qas[batch_start: batch_start + bsz]
                
                # check cache and filter new inputs
                new_batch = []
                for b in batch:
                    question = b['question']
                    new_evidence = []
                    new_se_pos = []
                    new_prediction = []
                    new_score = []
                    new_title = []
                    for evidence, se_pos, prediction, score, title in zip(b['evidence'], b['se_pos'], b['prediction'], b['score'], b['title']):
                        has_cache = self.ce_cache.is_cached(question, evidence, se_pos[0], se_pos[1])
                        
                        if not has_cache:
                            new_evidence.append(evidence)
                            new_se_pos.append(se_pos)
                            new_prediction.append(prediction)
                            new_score.append(score)
                            new_title.append(title)
                    
                    new_b = copy.deepcopy(b)
                    new_b['evidence'] = new_evidence
                    new_b['se_pos'] = new_se_pos
                    new_b['prediction'] = new_prediction
                    new_b['score'] = new_score
                    new_b['title'] = new_title
                    if len(new_evidence)> 0:
                        new_batch.append(new_b)
                    
                if len(new_batch) != 0:
                    padded_batch = len(new_batch) == 1
                    if padded_batch: # hack for batch size 1 issues
                        new_batch = [new_batch[0],new_batch[0]]
                    
                    inputs = self.tokenize_for_reranker(new_batch, use_cuda)
                    logits = forward(inputs)
                    if padded_batch:
                        logits = logits[:-1]
                        new_batch = [new_batch[0]]
                    
                    # save scores to cache
                    for b, logit in zip(new_batch, logits):
                        question = b['question']
                        for evidence, se_pos, lg in zip(b['evidence'], b['se_pos'], logit):
                            self.ce_cache.set_cache(question, evidence, se_pos[0], se_pos[1], lg)
                        
                # restore stores
                scores = []
                for b in batch:
                    score = []
                    question = b['question']
                    for evidence, se_pos in zip(b['evidence'], b['se_pos']):
                        logit = self.ce_cache.get_cache(question, evidence, se_pos[0], se_pos[1])
                        score.append(logit[1])
                    score = score[:rerank_k]
                    scores.append(score)
                scores = torch.Tensor(scores)
                            
                scores = torch.nn.Softmax(dim=-1)(scores)
                
                # aggregate densephrases score
                if rerank_lambda < 1:
                    dph_scores = [b['score'][:rerank_k][:len(b['prediction'])] for b in batch]
                    dph_scores = [[0 if _ == '' else _ for _ in d] for d in dph_scores]
                    dph_scores = [[_/100 for _ in d] for d in dph_scores] # for smoothing softmax 
                    dph_scores = torch.nn.Softmax(dim=-1)(torch.tensor(dph_scores))
                    scores = rerank_lambda * scores + (1 - rerank_lambda) * dph_scores
                
                inds = torch.argsort(scores, descending=True)
                
                outputs.extend(inds.tolist())
                output_scores.extend(scores.tolist())

        return get_output_format(qas, outputs, output_scores)