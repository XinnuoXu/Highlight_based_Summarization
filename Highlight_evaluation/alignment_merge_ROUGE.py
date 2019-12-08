#coding=utf8

import json
import sys
import os
import numpy as np
from evaluation_metrics import *
from scipy.stats import pearsonr
from scipy import spatial
sys.path.append(os.path.abspath('./'))
from alignment_merge_check import get_ground_truth, get_prediction, correlation

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

def load_gold(gold_highlight_path, doc_trees, task):
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

    gtruths = {}
    for doc_id in gold_highlight: 
        article_lst = doc_trees[doc_id]
        gtruth = get_ground_truth(article_lst, gold_highlight[doc_id], task)
        gtruths[doc_id] = gtruth

    return gtruths

def load_auto_alg_simple_format(prediction_path, task):
    preds = {}
    for line in open(prediction_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj['doc_id']

        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        #doc = _g2g_token_replace(article_lst)

        pred = get_prediction(attn_dists, article_lst, decoded_lst, task)
        preds[doc_id] = pred
    return preds

def load_auto_alg(prediction_path, task):
    preds = {}
    for filename in os.listdir(prediction_path):
        with open(prediction_path + "/" + filename, 'r') as file:
            json_obj = json.loads(file.read().strip())
        doc_id = filename.split(".")[0]

        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        #doc = _g2g_token_replace(article_lst)

        pred = get_prediction(attn_dists, article_lst, decoded_lst, task)
        preds[doc_id] = pred
    return preds

def load_doc_trees(doc_tree_path):
    doc_trees = {}
    for filename in os.listdir(doc_tree_path):
        if filename.endswith("tgt"):
            continue
        doc_id = filename.split(".")[0]
        with open(doc_tree_path + filename, 'r') as file:
            doc_trees[doc_id] = file.read().strip().split()
    return doc_trees

def evaluation(gold_highlight, pred_highlight):
    corr_all = []; corrs_01 = []
    for doc_id in gold_highlight:
        gtruth = gold_highlight[doc_id]
        pred = pred_highlight[doc_id]
        corr, corr_01 = correlation(gtruth, pred, doc_id)
        if corr > -2:
            corr_all.append(corr)
            corrs_01.append(corr_01)

    return sum(corr_all)/len(corr_all), sum(corrs_01)/len(corrs_01)

if __name__ == '__main__':
    # Load ground truth
    if sys.argv[1] == "human":
        doc_trees = load_doc_trees("50_trees/")
        gold_phrase_path = "AMT_data/alignment_phrase.jsonl"
        gold_highlight_phrase = load_gold(gold_phrase_path, doc_trees, "phrase")
        gold_fact_path = "AMT_data/alignment_fact.jsonl"
        gold_highlight_fact = load_gold(gold_fact_path, doc_trees, "fact")
    elif sys.argv[1] == "system":
        prediction_path = "Bert_highlight/"
        gold_highlight_phrase = load_auto_alg(prediction_path, "phrase")
        gold_highlight_fact = load_auto_alg(prediction_path, "fact")
    elif sys.argv[1] == "auto_full":
        prediction_path = "system_trees/" + sys.argv[2] + "_gold.alg"
        gold_highlight_phrase = load_auto_alg_simple_format(prediction_path, "phrase")
        gold_highlight_fact = load_auto_alg_simple_format(prediction_path, "fact")

    # Load system alignment
    if sys.argv[1] == "auto_full":
        prediction_path = "system_trees/" + sys.argv[2] + "_full.alg"
        highlight_phrase = load_auto_alg_simple_format(prediction_path, "phrase")
        highlight_fact = load_auto_alg_simple_format(prediction_path, "fact")
    else:
        prediction_path = "system_trees/" + sys.argv[2]
        highlight_phrase = load_auto_alg(prediction_path, "phrase")
        highlight_fact = load_auto_alg(prediction_path, "fact")
        

    # Calculate correlation
    fact_merge, fact_01 = evaluation(gold_highlight_fact, highlight_fact)
    phrase_merge, phrase_01 = evaluation(gold_highlight_phrase, highlight_phrase)

    #print ("fact_single, fact_merge ", fact_single, fact_merge)
    #print ("phrase_single, phrase_merge ", phrase_single, phrase_merge)
    print ("fact", fact_merge, fact_01)
    print ("phrase", phrase_merge, phrase_01)
