#coding=utf8

import json
import sys
import os
import numpy as np
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
    #threshold = max(np.sort(tok_align)[-1] * 0.6, np.sort(tok_align)[-n])
    threshold = np.sort(tok_align)[-1] * 0.3
    for i in range(len(tok_align)):
        if tok_align[i] < threshold:
            tok_align[i] = 0.0
    return tok_align

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

def get_ground_truth(decoded_lst, gold_human_label):
    ret_scores = {}
    doc = []
    doc_len = 0
    for tag in gold_human_label:
        json_obj = gold_human_label[tag]
        doc = json_obj["document"]
        uni_gram_scores = json_obj["uni_gram_scores"]
        ret_scores[tag] = uni_gram_scores
        doc_len = len(uni_gram_scores)

    attn_dists = []; p_gens = []
    for tok in decoded_lst:
        tok = tok[1:]
        if tok in ret_scores:
            attn_dists.append(ret_scores[tok])
            p_gens.append(max(ret_scores[tok]))
        else:
            attn_dists.append([0.0]*doc_len)
            p_gens.append(0.0)
    return doc, attn_dists, p_gens

def get_ground_truth_phrase(decoded_lst, gold_human_label):
    ret_scores = {}
    doc = []
    doc_len = 0
    for tag in gold_human_label:
        json_obj = gold_human_label[tag]
        doc = json_obj["document"]
        uni_gram_scores = json_obj["uni_gram_scores"]
        ret_scores[tag] = uni_gram_scores
        doc_len = len(uni_gram_scores)

    attn_dists = []; p_gens = []; fact_stack = []; label_stack = []
    for tok in decoded_lst:
        cls = _label_classify(tok)
        tok = tok[1:]
        if cls == "fact":
            fact_stack.append(tok)
            label_stack.append(cls)
        elif cls == "phrase":
            tok = fact_stack[-1] + "|||" + tok
            label_stack.append(cls)
        elif cls == "end":
            if label_stack.pop() == "fact":
                fact_stack.pop()

        if tok in ret_scores:
            attn_dists.append(ret_scores[tok])
            p_gens.append(max(ret_scores[tok]))
        else:
            attn_dists.append([0.0]*doc_len)
            p_gens.append(0.0)
    return doc, attn_dists, p_gens

if __name__ == '__main__':
    prediction_path = "Bert_highlight/"
    human_path = "Human_highlight/"
    if sys.argv[1] == "phrase":
        gold_highlight_path = "AMT_data/alignment_phrase.jsonl"
    elif sys.argv[1] == "fact":
        gold_highlight_path = "AMT_data/alignment_fact.jsonl"
    gold_highlight = load_gold(gold_highlight_path)

    for filename in os.listdir(prediction_path):
        with open(prediction_path + filename, 'r') as file:
            json_obj = json.loads(file.read().strip())

        doc_id = filename.split(".")[0]
        decoded_lst = json_obj['decoded_lst']
        gold_human_label = gold_highlight[doc_id]
        if sys.argv[1] == "phrase":
            doc, attn_dists, p_gens = get_ground_truth_phrase(decoded_lst, gold_human_label)
        else:
            doc, attn_dists, p_gens = get_ground_truth(decoded_lst, gold_human_label)


        json_obj['article_lst'] = doc
        json_obj['p_gens'] = p_gens
        json_obj['attn_dists'] = attn_dists
        json_obj["abstract_str"] = "..."

        fpout = open(human_path + doc_id + ".hl", "w")
        fpout.write(json.dumps(json_obj) + "\n")
        fpout.close()
