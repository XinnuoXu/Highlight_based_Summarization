#coding=utf8

import sys, json
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
    tokens = []; scores = []
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
        else:
            tokens.append(token)
            label_score = label_score_stack[-1] if len(label_score_stack) > 0 else 0.0
            fact_score = fact_score_stack[-1] if len(fact_score_stack) > 0 else 0.0
            score = ALPHA_PH * label_score + ALPHA_FA * fact_score
            scores.append(score)
    return tokens, scores

def _document(article_lst, attn_dists):
    attn_dists = np.array(attn_dists)
    sum_dists = np.amax(attn_dists, axis=0)
    return _re_score(article_lst, sum_dists)

def _summary(decoded_lst, p_gens):
    return _re_score(decoded_lst, p_gens)

if __name__ == '__main__':
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
