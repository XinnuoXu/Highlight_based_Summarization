#coding=utf8

import json
import sys
import os
from evaluation_metrics import *
from scipy.stats import pearsonr

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
    threshold = np.sort(tok_align)[-n]
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
    ret_scores = {}; type_stuck = []; fact_stuck = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            type_stuck.append(cls)
            fact_stuck.append(tok)
        elif cls == "phrase":
            type_stuck.append(cls)
            tag = fact_stuck[-1][1:] + "|||" + tok[1:]
            ret_scores[tag] = _filter_attn(article_lst, attn_dists[i], "phrase")
        elif cls == "end":
            if type_stuck.pop() == "fact":
                fact_stuck.pop()
    return ret_scores

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
    ret_scores = {}
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            tag = tok[1:]
            ret_scores[tag] = _filter_attn(article_lst, attn_dists[i], "fact")
    return ret_scores

def get_ground_truth(doc, gold_human_label, task):
    if task == "fact":
        phrase_to_tokens = _fact_to_tokens(doc)
    elif task == "phrase":
        phrase_to_tokens = _phrase_to_tokens(doc)
    ret_scores = {}
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
        ret_scores[tag] = ph_scores
    return ret_scores

def get_prediction(attn_dists, article_lst, decoded_lst, task):
    if task == "phrase":
        return _prediction_phrase(article_lst, attn_dists, decoded_lst)
    elif task == "fact":
        return _prediction_fact(article_lst, attn_dists, decoded_lst)
    else:
        return _prediction_token(article_lst, attn_dists, decoded_lst)

def correlation(gtruth, pred, doc_id):
    corrs = []
    for item in pred:
        if item not in gtruth:
            continue
        if len(gtruth[item]) == 0 or len(pred[item]) == 0:
            continue
        if sum(gtruth[item]) == 0 or sum(pred[item]) == 0:
            continue
        g = gtruth[item]
        p = _top_n_filter(pred[item], 10)
        print (doc_id, item)
        print ("g", g)
        print ("p", p)
        print (pearsonr(g, p)[0])
        corrs.append(pearsonr(g, p)[0])
    return corrs

if __name__ == '__main__':
    prediction_path = "Bert_highlight/"
    if sys.argv[1] == "phrase":
        gold_highlight_path = "AMT_data/alignment_phrase.jsonl"
    elif sys.argv[1] == "fact":
        gold_highlight_path = "AMT_data/alignment_fact.jsonl"

    gold_highlight = load_gold(gold_highlight_path)

    corrs = []
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
        corrs.extend(correlation(gtruth, pred, doc_id))
    print (corrs)
    print (sum(corrs)/len(corrs))

