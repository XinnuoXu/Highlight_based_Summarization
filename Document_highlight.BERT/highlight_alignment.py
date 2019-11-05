#coding=utf8

import sys, json, os
import numpy as np
#from multiprocess import Pool
import multiprocessing

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
            if len(type_stack) > 0 and type_stack[-1] == "phrase" and len(label_score_stack) > 0:
                score += label_score_stack[-1]
            if len(fact_score_stack) > 0:
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
    decoded_lst, scores = _merge_weight(decoded_lst, attn_dists)
    scores.insert(0, sum_dists)
    decoded_lst.insert(0, "[SUMMARY]")
    p_gens.insert(0, 1.0)
    scores = [_re_score(article_lst, score) for score in scores]
    article_lst = _rephrase(article_lst)
    return article_lst, decoded_lst, scores

def _split_doc(article_lst, attn_dists, split_info):
    cut_pos = [0]
    for sen in split_info:
        cut_pos.append(cut_pos[-1] + len(sen))
    cut_attn_dists = [[] for item in attn_dists] 
    cut_attn_dists.append([])
    sen_idx = 0
    for i, tok in enumerate(article_lst):
        if i == cut_pos[sen_idx]:
            for j in range(len(cut_attn_dists)):
                cut_attn_dists[j].append([])
            sen_idx+=1
        for j in range(len(attn_dists)):
            cut_attn_dists[j][-1].append(attn_dists[j][i])
        cut_attn_dists[-1][-1].append(tok)
    return cut_attn_dists[-1], cut_attn_dists[:-1]

def _one_file(label):
    file_name = "./highlights.bert/xsum_" + label + ".jsonl"
    fpout_dir = "./highlights.bert/for_alignment/" + label + ".jsonl"
    fpout = open(fpout_dir, "w")

    for i, line in enumerate(open(file_name)):
        json_obj = json.loads(line.strip())
        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        doc = json_obj['ctx_trees']
        doc = [_rephrase(sen.split(' ')) for sen in doc]

        p_gens = _re_score(decoded_lst, p_gens)
        article_lst, decoded_lst, attn_dists = _alignment(article_lst, decoded_lst, attn_dists, p_gens)
        article_lst, attn_dists = _split_doc(article_lst, attn_dists, doc)

        json_obj = {}
        json_obj["article_lst"] = article_lst
        json_obj["decoded_lst"] = decoded_lst
        json_obj["abstract_str"] = abstract_str
        json_obj["attn_dists"] = attn_dists
        json_obj["p_gens"] = p_gens
        fpout.write(json.dumps(json_obj) + "\n")
        
        '''
        fpout = open("tmp_output.thres/" + str(i) + ".json", "w")
        fpout.write(json.dumps(json_obj) + "\n")
        fpout.close()
        '''

    fpout.close()

def g2g_token_replace(tokens):
    at_at_num = 0
    quote_num = 0
    for i in range(len(tokens)):
        if tokens[i] == "@@":
            at_at_num += 1
            if at_at_num % 2 == 1:
                tokens[i] = "-lrb-"
            else:
                tokens[i] = "-rrb-"
        elif tokens[i] == "£":
            tokens[i] = "#"
        elif tokens[i] == "\"":
            quote_num += 1
            if quote_num % 2 == 1:
                tokens[i] = "``"
            else:
                tokens[i] = "\'\'"
    return tokens

def preprocess(file_dir):
    objs = []
    for line in open(file_dir):
        json_obj = json.loads(line.strip())
        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        src = [g2g_token_replace(sen) for sen in article_lst]
        tgt = g2g_token_replace(decoded_lst)

        obj = {}
        obj['src'] = src
        obj['tgt'] = tgt
        obj['alignment'] = attn_dists
        objs.append(obj)
    return objs

if __name__ == '__main__':
    if sys.argv[1] == "onefile":
        _one_file(sys.argv[2])
    if sys.argv[1] == "multi_thread":
        for filename in os.listdir("/scratch/xxu/highlights/"):
            label = filename.replace("xsum_", "").replace(".jsonl", "")
            os.system("nohup python highlight_alignment.py onefile " + label + " &")
    if sys.argv[1] == "merge":
        datasets = {}
        for filename in os.listdir("/scratch/xxu/highlights/for_alignment/"):
            flist = filename.split(".")[0].split("_")
            if len(flist) == 2:
                continue
            tag = flist[0]
            if tag not in datasets:
                datasets[tag] = []
            file_dir = "/scratch/xxu/highlights/for_alignment/" + filename
            data_list = preprocess(file_dir)
            datasets[tag].extend(data_list)
        for tag in datasets:
            fpout = open("/scratch/xxu/highlights/s2s_alignment/xsum_" + tag + "_src.jsonl", "w")
            for obj in datasets[tag]:
                fpout.write(json.dumps(obj) + "\n")
            fpout.close()
