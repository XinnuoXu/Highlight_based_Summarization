#coding=utf8

import json
import sys
import os
import numpy as np
from evaluation_metrics import *
from scipy.stats import pearsonr
from scipy import spatial
sys.path.append(os.path.abspath('../Document_highlight.BERT/'))
from highlight import phrase_attn_to_fact

def _label_classify(item):
    if item[0] == '(':
        if item[1] == 'F':
            return "fact"
        else:
            return "phrase"
    elif item[0] == ')':
        return "end"
    elif item[0] == '*':
        return "reference"
    return "token"

def _top_n_filter(tok_align, n):
    if n == -1:
        return tok_align
    #threshold = max(np.sort(tok_align)[-1] * 0.6, np.sort(tok_align)[-n])
    threshold = np.sort(tok_align)[-1] * 0.3
    for i in range(len(tok_align)):
        if tok_align[i] < threshold:
            tok_align[i] = 0.0
    return tok_align

def _g2g_token_replace(tokens):
    at_at_num = 0
    for i in range(len(tokens)):
        if tokens[i] == "@@":
            at_at_num += 1
            if at_at_num % 2 == 1:
                tokens[i] = "("
            else:
                tokens[i] = ")"
        elif tokens[i] == "£":
            tokens[i] = "#"
        elif tokens[i] == "[":
            tokens[i] = "-lsb-"
        elif tokens[i] == "]":
            tokens[i] = "-rsb-"
    return tokens

def load_gold(gold_highlight_path):
    gold_highlight = {}
    for line in open(gold_highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        summ = json_obj["summary"]
        if isinstance(summ, dict):
            tag = summ["name"]
            if doc_id not in gold_highlight:
                gold_highlight[doc_id] = {}
            gold_highlight[doc_id][tag] = json_obj
        else:
            gold_highlight[doc_id] = json_obj
    return gold_highlight

def _phrase_to_tokens(article_lst):
    phrase_dict = []; phrase_idx = []
    type_stack = []
    counter = {}
    for i, token in enumerate(article_lst):
        token_type = _label_classify(token)
        if token_type == "fact":
            type_stack.append(token_type)
        elif token_type == "phrase":
            type_stack.append(token_type)
            phrase_idx.append(len(phrase_dict))
            phrase_dict.append([])
        elif token_type == "end":
            pop_type = type_stack.pop()
            if pop_type == "phrase":
                phrase_idx.pop()
        elif token_type != "reference":
            if token not in counter:
                counter[token] = 0
            else:
                counter[token] += 1
            pos = counter[token]
            token = str(pos) + "-" + token
            if len(type_stack) > 0 and type_stack[-1] == "phrase":
                phrase_dict[phrase_idx[-1]].append(token)
    return phrase_dict

def _filter_attn(article_lst, attn, task):
    phrase_attn = []
    for i, token in enumerate(article_lst):
        token_type = _label_classify(token)
        if token_type == task:
            phrase_attn.append(attn[i])
    return phrase_attn

def _prediction_phrase(article_lst, attn_dists, decoded_lst):
    ret_scores = []; type_stuck = []; fact_stuck = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            type_stuck.append(cls)
            fact_stuck.append(tok)
        elif cls == "phrase":
            type_stuck.append(cls)
            tag = fact_stuck[-1][1:] + "|||" + tok[1:]
            ret_scores.append(np.array(_filter_attn(article_lst, attn_dists[i], "phrase")))
        elif cls == "end":
            if type_stuck.pop() == "fact":
                fact_stuck.pop()
    if len(ret_scores) == 0:
        return []
    ret_scores = sum(ret_scores) / len(ret_scores)
    return ret_scores.tolist()

def _fact_to_tokens(article_lst):
    fact_dict = []; fact_idx = []
    type_stack = []
    counter = {}
    for i, token in enumerate(article_lst):
        token_type = _label_classify(token)
        if token_type == "fact":
            type_stack.append(token_type)
            fact_idx.append(len(fact_dict))
            fact_dict.append([])
        elif token_type == "phrase":
            type_stack.append(token_type)
        elif token_type == "end":
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_idx.pop()
        elif token_type != "reference":
            if token not in counter:
                counter[token] = 0
            else:
                counter[token] += 1
            pos = counter[token]
            token = str(pos) + "-" + token
            for j in range(len(fact_idx)):
                fact_dict[fact_idx[j]].append(token)
            #if len(fact_idx) > 0:
            #    fact_dict[fact_idx[-1]].append(token)
    return fact_dict

def _prediction_fact(article_lst, attn_dists, decoded_lst):
    ret_scores = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            tag = tok[1:]
            ret_scores.append(np.array(_filter_attn(article_lst, attn_dists[i], "fact")))
    ret_scores = sum(ret_scores) / len(ret_scores)
    return ret_scores.tolist()

def get_ground_truth(doc_tree, gold_human_label, task):
    if task == "phrase":
        phrase_weights = get_ground_truth_phrase(doc_tree, gold_human_label)
        return phrase_weights
    elif task == "fact":
        phrase_weights = get_ground_truth_phrase(doc_tree, gold_human_label)
        fact_weights = get_ground_truth_fact(phrase_weights, doc_tree)
        return fact_weights

def map_phrase_to_all_tok(ph_score, doc):
    new_score = []
    ph_idx = 0
    for tok in doc:
        cls = _label_classify(tok)
        if cls == "phrase":
            new_score.append(ph_score[ph_idx])
            ph_idx += 1
        else:
            new_score.append(0.0)
    return new_score

def map_all_tok_to_fact(attn, doc):
    new_score = []
    for i, tok in enumerate(doc):
        cls = _label_classify(tok)
        if cls == "fact":
            new_score.append(attn[i])
    return new_score

def get_ground_truth_fact(phrase_to_tokens, doc):
    ret_scores = {}
    ph_score = map_phrase_to_all_tok(phrase_to_tokens, doc)
    ph_score = phrase_attn_to_fact(ph_score, doc)
    return map_all_tok_to_fact(ph_score, doc)

def get_ground_truth_phrase(doc, gold_human_label):
    phrase_to_tokens = _phrase_to_tokens(doc)
    phrase_scores = []
    for tag in gold_human_label:
        json_obj = gold_human_label[tag]
        uni_gram = json_obj["uni_gram"]
        uni_gram_scores = json_obj["uni_gram_scores"]
        tok_scores = {}
        for i, item in enumerate(uni_gram):
            tok_scores[item] = uni_gram_scores[i]
        ph_scores = []
        for ph in phrase_to_tokens:
            score = 0.0
            for tok in ph:
                if tok not in tok_scores:
                    continue
                score += tok_scores[tok]
            if len(ph) == 0:
                ph_scores.append(0.0)
            else:
                ph_scores.append(score/len(ph))
        phrase_scores.append(np.array(ph_scores))
    phrase_scores = (sum(phrase_scores) / len(phrase_scores)).tolist()
    return phrase_scores

def get_prediction(attn_dists, article_lst, decoded_lst, task):
    if task == "phrase":
        return _prediction_phrase(article_lst, attn_dists, decoded_lst)
    elif task == "fact":
        return _prediction_fact(article_lst, attn_dists, decoded_lst)
    else:
        return _prediction_token(article_lst, attn_dists, decoded_lst)

def true_false(g, p):
    g_ = np.array(g)
    p_ = np.array(p)
    return 1 - spatial.distance.cosine(g_ > 0 , p_ > 0.1)

def correlation(gtruth, pred, doc_id):
    if len(gtruth) < 2 or len(pred) < 2:
        return -2, -2
    if sum(gtruth) == 0 or sum(pred) == 0:
        return -2, -2
    return pearsonr(gtruth, pred)[0], true_false(gtruth, pred)

if __name__ == '__main__':
    prediction_path = "Bert_highlight/"
    if sys.argv[1] == "phrase":
        gold_highlight_path = "AMT_data/alignment_phrase.jsonl"
    elif sys.argv[1] == "fact":
        gold_highlight_path = "AMT_data/alignment_fact.jsonl"

    gold_highlight = load_gold(gold_highlight_path)

    corrs = []; corr_all = []; ids = []; corrs_01 = []
    for filename in os.listdir(prediction_path):
        with open(prediction_path + filename, 'r') as file:
            json_obj = json.loads(file.read().strip())
        doc_id = filename.split(".")[0]

        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        #doc = _g2g_token_replace(article_lst)

        gold_human_label = gold_highlight[doc_id]
        gtruth = get_ground_truth(article_lst, gold_human_label, sys.argv[1])
        pred = get_prediction(attn_dists, article_lst, decoded_lst, sys.argv[1])
        corr, corr_01 = correlation(gtruth, pred, doc_id)
        if corr > -2:
            corr_all.append(corr)
            corrs_01.append(corr_01)
            ids.append(doc_id)

    #print (corrs)
    #print (corr_all)
    #for i, item in enumerate(corr_all):
    #    print (ids[i], item)
    print (sum(corr_all)/len(corr_all))
    print (sum(corrs_01)/len(corrs_01))

