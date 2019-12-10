#coding=utf8

import json
import sys
import os
import numpy as np
from evaluation_metrics import *
from scipy.stats import pearsonr
from scipy import spatial
sys.path.append(os.path.abspath('./'))
from alignment_check import get_ground_truth

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

def _prediction_fact(article_lst, attn_dists, decoded_lst):
    ret_scores = {}
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            tag = tok[1:]
            ret_scores[tag] = _filter_attn(article_lst, attn_dists[i], "fact")
    return ret_scores

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
    return 1 - spatial.distance.cosine(g_ > 0 , p_ > 0.2)

def correlation(gtruth, pred, doc_id):
    corrs = []
    p_wight = []; g_weight = []
    for item in pred:
        p_wight.append(np.array(pred[item]))
    if len(pred) == 0:
        return -2, -2
    pred = sum(p_wight) / len(pred)
    for item in gtruth:
        g_weight.append(np.array(gtruth[item]))
    gtruth = sum(g_weight) / len(gtruth)
    if sum(pred) == 0:
        return -2, -2
    if sum(gtruth) == 0:
        return -2, -2
    #return pearsonr(pred, gtruth)[0], true_false(pred, gtruth)
    return pearsonr(pred.tolist(), gtruth.tolist())[0], 0.0

def load_auto_alg_simple_format(prediction_path, task):
    preds = {};
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
    preds = {}; decode_list = {}
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
        decode_list[doc_id] = ' '.join(decoded_lst)
    return preds, decode_list

def load_doc_trees(doc_tree_path):
    doc_trees = {}
    for filename in os.listdir(doc_tree_path):
        if filename.endswith("tgt"):
            continue
        doc_id = filename.split(".")[0]
        with open(doc_tree_path + filename, 'r') as file:
            doc_trees[doc_id] = file.read().strip().split()
    return doc_trees

def eva_debug(gold_highlight, pred_highlight):
    corr_all = {}
    for doc_id in gold_highlight:
        gtruth = gold_highlight[doc_id]
        pred = pred_highlight[doc_id]
        corr, corr_01 = correlation(gtruth, pred, doc_id)
        if corr > -2:
            corr_all[doc_id] = corr
    return corr_all

if __name__ == '__main__':
    # Load human highlight
    doc_trees = load_doc_trees("50_trees/")
    gold_phrase_path = "AMT_data/alignment_phrase.jsonl"
    human_highlight_phrase = load_gold(gold_phrase_path, doc_trees, "phrase")
    gold_fact_path = "AMT_data/alignment_fact.jsonl"
    human_highlight_fact = load_gold(gold_fact_path, doc_trees, "fact")

    # Load gold summary highlight
    prediction_path = "Bert_highlight/"
    gold_highlight_phrase, gold_decode_list = load_auto_alg(prediction_path, "phrase")
    gold_highlight_fact, _ = load_auto_alg(prediction_path, "fact")

    systems = ["bert", "bertalg", "ptgen", "tconvs2s"]
    systems_res = {}
    for s in systems:
        # Load system alignment
        prediction_path = "system_trees/system_" + s + ".alg"
        highlight_phrase, system_docode_list = load_auto_alg(prediction_path, "phrase")
        highlight_fact,_  = load_auto_alg(prediction_path, "fact")

        # Calculate correlation
        fact_weight = eva_debug(gold_highlight_fact, highlight_fact)
        phrase_weight = eva_debug(gold_highlight_phrase, highlight_phrase)

        systems_res[s] = (system_docode_list, fact_weight, phrase_weight)

    fpout = open("system_analysis.res", "w")
    for doc_id in gold_decode_list:
        outputlist = []
        for s in systems_res:
            if doc_id not in systems_res[s][0] or \
                    doc_id not in systems_res[s][1] or \
                    doc_id not in systems_res[s][2]:
                        break
            decode_list = systems_res[s][0][doc_id]
            fact_weight = str(systems_res[s][1][doc_id])
            phrase_weight = str(systems_res[s][2][doc_id])
            outputs = [doc_id, s.upper(), decode_list, fact_weight, phrase_weight]
            outputlist.append(outputs)
        if len(outputlist) == len(systems_res):
            golds = [doc_id, "GOLD", gold_decode_list[doc_id]]
            fpout.write("\t".join(golds) + "\n")
            for outputs in outputlist:
                fpout.write("\t".join(outputs) + "\n")
        fpout.write("\n")
    fpout.close()
