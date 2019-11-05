#coding=utf8

import json
import sys
import os
from evaluation_metrics import *

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

def hardy_statistic(toks, scores):
    non_zeros = 0
    for s in scores:
        if s > 0.0:
            non_zeros += 1
    return non_zeros

class Eva():
    def __init__(self):
        self.ndcg_metrics = [10, 30, 50, 100, 150]
        self.ndcg_score = [0.0] * len(self.ndcg_metrics)

        self.rec_metrics = [10, 30, 50, 100, 150]
        self.rec_score = [0.0] * len(self.rec_metrics)
        self.prec_metrics = [10, 20]
        self.prec_score = [0.0] * len(self.prec_metrics)
        self.rec_prec_threshold = 50

        self.doc_num = 0
        self.regardless_pos = False

    def no_pos(self, toks, scores):
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

    def ndcg(self, h_toks, h_scores, my_toks, my_scores):
        h_dict = {}
        for i, item in enumerate(h_toks):
            h_dict[item] = h_scores[i]
        tmp_dict = {}
        for i, item in enumerate(my_toks):
            tmp_dict[item] = my_scores[i]
        r = []
        for item in sorted(tmp_dict.items(), key = lambda d:d[1], reverse=True):
            tok = item[0]
            if item[0] not in h_dict:
                continue
            r.append(h_dict[item[0]])
        for i, k in enumerate(self.ndcg_metrics):
            res = ndcg_at_k(r, k)
            self.ndcg_score[i] += res

    def rec_prec(self, h_toks, h_scores, my_toks, my_scores):
        threshold = sorted(h_scores, reverse=True)[self.rec_prec_threshold]
        threshold = max(0.2, threshold)

        h_dict = {}
        for i, item in enumerate(h_toks):
            if h_scores[i] > threshold:
                h_dict[item] = 1
            else:
                h_dict[item] = 0
        tmp_dict = {}
        for i, item in enumerate(my_toks):
            tmp_dict[item] = my_scores[i]
        r = []
        for item in sorted(tmp_dict.items(), key = lambda d:d[1], reverse=True):
            tok = item[0]
            if item[0] not in h_dict:
                continue
            r.append(h_dict[item[0]])
        for i, k in enumerate(self.rec_metrics):
            rec_res = recall_at_k(r, k)
            self.rec_score[i] += rec_res
            if k == 20:
                print (rec_res)

    def one_doc(self, h_toks, h_scores, my_toks, my_scores):
        self.doc_num += 1
        if self.regardless_pos:
            my_toks, my_scores = self.no_pos(my_toks, my_scores)
            h_toks, h_scores = self.no_pos(h_toks, h_scores)

        self.ndcg(h_toks, h_scores, my_toks, my_scores)
        self.rec_prec(h_toks, h_scores, my_toks, my_scores)

    def report_res(self):
        print ("ndcg@k", [item / self.doc_num for item in self.ndcg_score])
        print ("rec@k", [item / self.doc_num for item in self.rec_score])

if __name__ == '__main__':
    token_weight_path = "Bert_token_weight/"
    #token_weight_path = "GloVE_token_weight/"
    hardy_highlight_path = "./Hardy_HROUGE/highres/highlight.jsonl"
    hardy_highlight = {}
    for line in open(hardy_highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        hardy_highlight[doc_id] = json_obj

    uni_eva = Eva()
    bi_eva = Eva()
    tri_eva = Eva()
    qua_eva = Eva()

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
        h_uni_gram = hardy_highlight[doc_id]["uni_gram"]
        h_uni_gram_scores = hardy_highlight[doc_id]["uni_gram_scores"]

        '''
        # BiGram
        '''
        h_bi_gram = hardy_highlight[doc_id]["bi_gram"]
        h_bi_gram_scores = hardy_highlight[doc_id]["bi_gram_scores"]

        '''
        # TriGram
        '''
        h_tri_gram = hardy_highlight[doc_id]["tri_gram"]
        h_tri_gram_scores = hardy_highlight[doc_id]["tri_gram_scores"]

        '''
        # QuaGram
        '''
        h_qua_gram = hardy_highlight[doc_id]["qua_gram"]
        h_qua_gram_scores = hardy_highlight[doc_id]["qua_gram_scores"]

        uni_eva.one_doc(h_uni_gram, h_uni_gram_scores, uni_gram, uni_gram_scores)
        bi_eva.one_doc(h_bi_gram, h_bi_gram_scores, bi_gram, bi_gram_scores)
        tri_eva.one_doc(h_bi_gram, h_bi_gram_scores, bi_gram, bi_gram_scores)
        qua_eva.one_doc(h_qua_gram, h_qua_gram_scores, qua_gram, qua_gram_scores)

    print ("Uni:")
    print (uni_eva.report_res())
    print ("Bi:")
    print (bi_eva.report_res())
    print ("Tri:")
    print (tri_eva.report_res())
    print ("Qua:")
    print (qua_eva.report_res())
