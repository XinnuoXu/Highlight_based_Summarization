#coding=utf8

import json
import sys
import os
from evaluation_metrics import *
from scipy.stats import pearsonr

def g2g_token_replace(tokens):
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

def ngram(n, D, H):
    m = len(D)
    score_list = []
    phrase_list = []
    idx_dict = {}
    for i in range(m-n+1):
        phrase = " ".join(D[i:i+n])
        if phrase not in idx_dict:
            idx_dict[phrase] = 0
        else:
            idx_dict[phrase] += 1
        phrase_list.append(str(idx_dict[phrase]) + "-" + phrase)
        total_NumH = 0
        for j in range(i, i+n):
            total_NumH += H[i] / n
        score_list.append(total_NumH)
    return phrase_list, score_list

def no_pos(toks, scores):
    sdict = {}; fdict = {}
    for i, tok in enumerate(toks):
        tok = '-'.join(tok.split('-')[1:])
        if tok not in sdict:
            sdict[tok] = scores[i]
            fdict[tok] = 1
        else:
            sdict[tok] += scores[i]
            fdict[tok] += 1
    for tok in sdict:
        sdict[tok] = sdict[tok] / fdict[tok]
    return list(sdict.keys()), list(sdict.values())

def preprocess(h_toks, h_scores, my_toks, my_scores, regardless_pos=False):
    if regardless_pos:
        my_toks, my_scores = no_pos(my_toks, my_scores)
        h_toks, h_scores = no_pos(h_toks, h_scores)

    s_dict = {}
    for i, item in enumerate(h_toks):
        if item in my_toks:
            s_dict[item] = [h_scores[i], 0]
    for i, item in enumerate(my_toks):
        if item in s_dict:
            s_dict[item][1] = my_scores[i]

    x = [s_dict[item][0] for item in s_dict]
    y = [s_dict[item][1] for item in s_dict]

    return pearsonr(x, y)

def highlight_corr():
    token_weight_path = "Bert_token_weight/"
    gold_highlight_path = "./Hardy_HROUGE/highres/highlight.jsonl"
    gold_highlight = {}
    for line in open(gold_highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        gold_highlight[doc_id] = json_obj

    uni_cs = []; uni_ps = []
    bi_cs = []; bi_ps = []
    tri_cs = []; tri_ps = []
    qua_cs = []; qua_ps = []

    for filename in os.listdir(token_weight_path):
        with open(token_weight_path + filename, 'r') as file:
            json_obj = json.loads(file.read().strip())
        doc_id = filename.split(".")[0]
        doc = g2g_token_replace(json_obj["article_lst"])
        attn_dists = json_obj["attn_dists"][0]

        uni_gram, uni_gram_scores = ngram(1, doc, attn_dists)
        bi_gram, bi_gram_scores = ngram(2, doc, attn_dists)
        tri_gram, tri_gram_scores = ngram(3, doc, attn_dists)
        qua_gram, qua_gram_scores = ngram(4, doc, attn_dists)

        '''
        # UniGram
        '''
        h_uni_gram = gold_highlight[doc_id]["uni_gram"]
        h_uni_gram_scores = gold_highlight[doc_id]["uni_gram_scores"]

        '''
        # BiGram
        '''
        h_bi_gram = gold_highlight[doc_id]["bi_gram"]
        h_bi_gram_scores = gold_highlight[doc_id]["bi_gram_scores"]

        '''
        # TriGram
        '''
        h_tri_gram = gold_highlight[doc_id]["tri_gram"]
        h_tri_gram_scores = gold_highlight[doc_id]["tri_gram_scores"]

        '''
        # QuaGram
        '''
        h_qua_gram = gold_highlight[doc_id]["qua_gram"]
        h_qua_gram_scores = gold_highlight[doc_id]["qua_gram_scores"]

        uni_c, uni_p = preprocess(h_uni_gram, h_uni_gram_scores, uni_gram, uni_gram_scores)
        bi_c, bi_p = preprocess(h_bi_gram, h_bi_gram_scores, bi_gram, bi_gram_scores)
        tri_c, tri_p = preprocess(h_tri_gram, h_tri_gram_scores, tri_gram, tri_gram_scores)
        qua_c, qua_p = preprocess(h_qua_gram, h_qua_gram_scores, qua_gram, qua_gram_scores)

        uni_cs.append(uni_c); uni_ps.append(uni_p)
        bi_cs.append(bi_c); bi_ps.append(bi_p)
        tri_cs.append(tri_c); tri_ps.append(tri_p)
        qua_cs.append(qua_c); qua_ps.append(qua_p)

        label = json_obj["abstract_str"].replace(" ", "_")
        print (filename, label, uni_c, bi_c, tri_c, qua_c)

    print ("Uni:")
    print (sum(uni_cs)/len(uni_cs), sum(uni_ps)/len(uni_ps))
    print ("Bi:")
    print (sum(bi_cs)/len(bi_cs), sum(bi_ps)/len(bi_ps))
    print ("Tri:")
    print (sum(tri_cs)/len(tri_cs), sum(tri_ps)/len(tri_ps))
    print ("Qua:")
    print (sum(qua_cs)/len(qua_cs), sum(qua_ps)/len(qua_ps))

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

req_names = [("uni_gram", "uni_gram_scores"),\
        ("bi_gram", "bi_gram_scores"),\
        ("tri_gram", "tri_gram_scores"),\
        ("qua_gram", "qua_gram_scores")]

def k_gram_corr(k, doc, attn_dists, gold_highlight):
    k_gram, k_gram_scores = ngram(k, doc, attn_dists)

    (req_gram, req_scores) = req_names[k-1]
    gold_k_gram = gold_highlight[req_gram]
    gold_k_gram_scores = gold_highlight[req_scores]

    return preprocess(gold_k_gram, gold_k_gram_scores, k_gram, k_gram_scores)

if __name__ == '__main__':
    if sys.argv[1] == "phrase":
        gold_highlight_path = "AMT_data/alignment_phrase.jsonl"
        token_weight_path = "Bert_phrase/"
    elif sys.argv[1] == "fact":
        gold_highlight_path = "AMT_data/alignment_fact.jsonl"
        token_weight_path = "Bert_fact/"

    gold_highlight = load_gold(gold_highlight_path)

    uni_cs = []; uni_ps = []
    bi_cs = []; bi_ps = []
    tri_cs = []; tri_ps = []
    qua_cs = []; qua_ps = []

    max_uni_cs = []; max_bi_cs = []; max_tri_cs = []; max_qua_cs = []
    for filename in os.listdir(token_weight_path):
        with open(token_weight_path + filename, 'r') as file:
            json_obj_list = json.loads(file.read().strip())

        doc_id = filename.split(".")[0]
        if not isinstance(json_obj_list, list):
            json_obj_list = [json_obj_list]

        tmp_uni = []; tmp_bi = []; tmp_tri = []; tmp_qua = []
        for json_obj in json_obj_list:
            doc = g2g_token_replace(json_obj["article_lst"])
            attn_dists = json_obj["attn_dists"][0]
            gold_doc = gold_highlight[doc_id]
            if not isinstance(gold_doc, dict):
                gold_h = gold_doc
            else:
                tag = json_obj["abstract_str"]
                if tag not in gold_doc:
                    continue
                gold_h = gold_doc[tag]

            # temp code about ARGM-TMP
            if sum(attn_dists)==0:
                continue
            uni_c, uni_p = k_gram_corr(1, doc, attn_dists, gold_h)
            bi_c, bi_p = k_gram_corr(2, doc, attn_dists, gold_h)
            tri_c, tri_p = k_gram_corr(3, doc, attn_dists, gold_h)
            qua_c, qua_p = k_gram_corr(4, doc, attn_dists, gold_h)

            uni_cs.append(uni_c); uni_ps.append(uni_p)
            bi_cs.append(bi_c); bi_ps.append(bi_p)
            tri_cs.append(tri_c); tri_ps.append(tri_p)
            qua_cs.append(qua_c); qua_ps.append(qua_p)

            tmp_uni.append(uni_c)
            tmp_bi.append(bi_c)
            tmp_tri.append(tri_c)
            tmp_qua.append(qua_c)

            print (filename, tag, uni_c, bi_c, tri_c, qua_c)

        try:
            max_uni_cs.append(max(tmp_uni))
            max_bi_cs.append(max(tmp_bi))
            max_tri_cs.append(max(tmp_tri))
            max_qua_cs.append(max(tmp_qua))
        except:
            pass

    print ("Uni:")
    print (sum(uni_cs)/len(uni_cs), sum(uni_ps)/len(uni_ps))
    print ("Bi:")
    print (sum(bi_cs)/len(bi_cs), sum(bi_ps)/len(bi_ps))
    print ("Tri:")
    print (sum(tri_cs)/len(tri_cs), sum(tri_ps)/len(tri_ps))
    print ("Qua:")
    print (sum(qua_cs)/len(qua_cs), sum(qua_ps)/len(qua_ps))

    print ("Uni max:")
    print (max_uni_cs)
    print (sum(max_uni_cs)/len(max_uni_cs))
    print ("Bi max:")
    print (sum(max_bi_cs)/len(max_bi_cs))
    print ("Tri max:")
    print (sum(max_tri_cs)/len(max_tri_cs))
    print ("Qua max:")
    print (sum(max_qua_cs)/len(max_qua_cs))
