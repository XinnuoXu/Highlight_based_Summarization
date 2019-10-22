#coding=utf8

import sys, json, os
import numpy as np

ALPHA_PH = 0.5
ALPHA_FA = 0.5

def label_classify(item):
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

def _re_score(article_lst, sum_dists):
    scores = []
    type_stack = []
    label_score_stack = []
    fact_score_stack = []
    for i, token in enumerate(article_lst):
        sim_score = sum_dists[i] if sum_dists[i] > 0 else 0.0
        token_type = label_classify(token)
        if token_type == "fact":
            type_stack.append(token_type)
            fact_score_stack.append(sim_score)
        elif token_type == "phrase":
            type_stack.append(token_type)
            label_score_stack.append(sim_score)
        elif token_type == "end":
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_score_stack.pop()
            elif pop_type == "phrase":
                label_score_stack.pop()
        elif token_type != "reference":
            if len(type_stack) == 0:
                scores.append(0)
            else:
                if type_stack[-1] == "phrase":
                    label_score = label_score_stack[-1] if len(label_score_stack) > 0 else 0.0
                else:
                    label_score = 0
                fact_score = fact_score_stack[-1] if len(fact_score_stack) > 0 else 0.0
                score = ALPHA_PH * label_score + ALPHA_FA * fact_score
                scores.append(score)
    return scores

def _merge_weight(decoded_lst, attn_dists):
    scores = []; tokens = []
    type_stack = []
    label_score_stack = []
    fact_score_stack = []
    for i, token in enumerate(decoded_lst):
        token_type = label_classify(token)
        if token_type == "fact":
            type_stack.append(token_type)
            fact_score_stack.append(attn_dists[i])
        elif token_type == "phrase":
            type_stack.append(token_type)
            label_score_stack.append(attn_dists[i])
        elif token_type == "end":
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_score_stack.pop()
            elif pop_type == "phrase":
                label_score_stack.pop()
        elif token_type != "reference":
            tokens.append(token)
            score = attn_dists[i]
            if type_stack[-1] == "phrase":
                score += label_score_stack[-1]
            score += fact_score_stack[-1]
            scores.append(score)
    return tokens, scores

def _rephrase(article_lst):
    tokens = []
    for i, token in enumerate(article_lst):
        token_type = label_classify(token)
        if token_type == "token":
            tokens.append(token)
    return tokens

def _alignment(article_lst, decoded_lst, attn_dists, p_gens):
    attn_dists = np.array(attn_dists)
    sum_dists = np.amax(attn_dists, axis=0)
    tokens, scores = _merge_weight(decoded_lst, attn_dists)
    scores.insert(0, sum_dists)
    tokens.insert(0, "[SUMMARY]")
    p_gens.insert(0, 1.0)
    scores = [_re_score(article_lst, score) for score in scores]
    article_lst = _rephrase(article_lst)
    print (len(article_lst))
    print (len(scores[0]))

def _summary(decoded_lst, p_gens):
    return _re_score(decoded_lst, p_gens)

def _one_file(label):
    sen_split_path = "./tmp_data/" + "corpus_g2g_" + label + ".txt"
    docs = []; summs = []
    for line in open(sen_split_path):
        flist = line.strip().split("\t")
        docs.append(flist[:-1])
        summs.append(flist[-1])
    file_name = "/scratch/xxu/highlights/xsum_" + label + ".jsonl"
    for i, line in enumerate(open(file_name)):
        json_obj = json.loads(line.strip())
        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']

        _alignment(article_lst, decoded_lst, attn_dists, p_gens)
        #p_gens = _summary(decoded_lst, p_gens)

if __name__ == '__main__':
    _one_file("111")
    #for filename in os.listdir("/scratch/xxu/highlights/"):
    #    label = filename.replace("xsum_", "").replace(".jsonl", "")
    #    _one_file(label)

    '''
    file_name = "/scratch/xxu/highlights/xsum_" + sys.argv[1] + ".jsonl"
    for i, line in enumerate(open(file_name)):
        json_obj = json.loads(line.strip())
        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        article_lst, doc_highlight = _document(article_lst, attn_dists)
        decoded_lst, p_gens = _summary(decoded_lst, p_gens)

        attn_dists = [np.zeros(len(article_lst)).tolist()] * len(decoded_lst)
        attn_dists.insert(0, doc_highlight)
        decoded_lst.insert(0, "[SUMMARY]")
        p_gens.insert(0, 1.0)

        fpout = open("tmp_output.thres/" + str(i) + ".json", "w")
        json_obj = {}
        json_obj["article_lst"] = article_lst
        json_obj["decoded_lst"] = decoded_lst
        json_obj["abstract_str"] = abstract_str
        json_obj["attn_dists"] = attn_dists
        json_obj["p_gens"] = p_gens
        fpout.write(json.dumps(json_obj) + "\n")
        fpout.close()
    '''
