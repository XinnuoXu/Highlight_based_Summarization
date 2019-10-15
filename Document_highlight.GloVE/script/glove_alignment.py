#coding=utf8

import sys
import os
import json
import math
import numpy as np
from glove import Glove
from glove import Corpus
from scipy.special import softmax
from scipy.spatial.distance import cosine

def split_data(label):
    contexts = []; summaries = []
    for line in open("../fact_data/g2g/xsum_" + label + "_src.jsonl"):
        contexts.append(line.strip())
    for line in open("../fact_data/g2g/xsum_" + label + "_tgt.jsonl"):
        summaries.append(line.strip())
    fpout = open("tmp", "w")
    for i in range(len(summaries)):
        if i % 5000 == 0:
            fpout.close()
            fid = str(i / 5000)
            fpout = open("tmp_data/corpus_g2g_" + label + "_" + fid + ".txt", "w")
        summary = summaries[i]
        context = contexts[i]
        fpout.write(context + "\t" + summary + "\n")
    fpout.close()
    os.system("rm tmp")
    
def read_data(label):
    fpout = open("./tmp_data/corpus_g2g_" + label + ".txt", "w")
    contexts = []; summaries = []
    for line in open("./tmp_data/xsum_" + label + "_src.jsonl"):
        contexts.append(line.strip())
    for line in open("./tmp_data/xsum_" + label + "_tgt.jsonl"):
        summaries.append(line.strip())
    for i in range(len(summaries)):
        summary = summaries[i]
        context = contexts[i]
        fpout.write(context + "\t" + summary + "\n")
    fpout.close()
    
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

def glove_one_sentence(line, stop_words, glove_model):
    embs = []
    embs_weight = []
    type_stack = []
    label_stack = []
    phrase_id_stack = []
    fact_id_stack = []
    for i, item in enumerate(line):
        l_type = label_classify(item)
        if l_type in ["fact", "phrase"]:
            type_stack.append(l_type)
            label_stack.append(item)
            if l_type == "fact":
                fact_id_stack.append(i)
            else:
                phrase_id_stack.append(i)
            embs.append([])
            embs_weight.append([])
        elif l_type == "end":
            embs.append([])
            embs_weight.append([])
            label_stack.pop()
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_id_stack.pop()
            elif pop_type == "phrase":
                phrase_id_stack.pop()
        else:
            if (item in stop_words) or (item not in glove_model.dictionary):
                # deal with reference
                embs.append([])
                embs_weight.append([])
            else:
                glv_emb = glove_model.word_vectors[glove_model.dictionary[item]]
                embs.append([glv_emb])
                embs_weight.append([1])
                if len(type_stack) == 0:
                    continue
                elif type_stack[-1] == "fact":
                    embs[fact_id_stack[-1]].append(glv_emb)
                    embs_weight[fact_id_stack[-1]].append(1)
                    for f_id in fact_id_stack[:-1]:
                        embs[f_id].append(glv_emb)
                        embs_weight[f_id].append(0.5)
                else:
                    if label_stack[-1] not in ['(ARGM-TMP']:
                        embs[phrase_id_stack[-1]].append(glv_emb)
                        embs_weight[phrase_id_stack[-1]].append(1)
                        embs[fact_id_stack[-1]].append(glv_emb)
                        if label_stack[-1] in ['(V']:
                            embs_weight[fact_id_stack[-1]].append(2)
                        else:
                            embs_weight[fact_id_stack[-1]].append(1)
                        for f_id in fact_id_stack[:-1]:
                            embs[f_id].append(glv_emb)
                            embs_weight[f_id].append(0.5)
    return embs, embs_weight

def flatten_vec(vec):
    res = []
    article = []
    article_index = []
    for i, llist in enumerate(vec):
        article.append(" ".join(llist))
        article_index.extend([i]*len(llist))
        res.extend(llist)
    return res, article, article_index

def set_2_set(s_emb, c_emb, s_emb_w, c_emb_w):
    attn_score = 0.0
    if len(c_emb) == 0:
        return attn_score
    for i, s_term in enumerate(s_emb):
        if sum(s_term) == 0:
            continue
        s_weight = s_emb_w[i]
        scores = [0]
        for j, c_term in enumerate(c_emb):
            c_weight = c_emb_w[j]
            if sum(c_term) == 0:
                scores.append(0)
            else:
                sim_score = max(0, 1-cosine(s_term, c_term))
                # Only keep very related pairs
                sim_score = sim_score if sim_score > 0.7 else 0
                scores.append(sim_score * s_weight * c_weight)
        term_score = sum(scores)
        attn_score += term_score
    return attn_score / len(s_emb)

def get_attn_dists(context_embs, summary_emb, context_emb_weight, summary_emb_weight, context, summary):
    attn_dists = []
    term_len = len(context_embs)
    for i, s_emb in enumerate(summary_emb):
        if len(s_emb) == 0:
            attn_dists.append(np.zeros(term_len).tolist())
            continue
        attn = []
        for j, c_emb in enumerate(context_embs):
            if label_classify(context[j]) != label_classify(summary[i]):
                attn.append(0)
            else:
                score = set_2_set(s_emb, c_emb, summary_emb_weight[i], context_emb_weight[j])
                attn.append(score)
        attn_dists.append(attn)
    return attn_dists

def fact_level_attn(attn_dists, context, summary):
    tmp_attn = []
    for item in attn_dists:
        t_attn = []
        for i, term in enumerate(context):
            if label_classify(term) == "fact":
                t_attn.append(item[i])
        tmp_attn.append(t_attn)
    default_len = len(tmp_attn[0])

    attn_dists = tmp_attn
    fact_attn_stack = []; label_stack = []; tokens = []
    for j, attn_f in enumerate(attn_dists):
        label_type = label_classify(summary[j])
        if label_type == "fact":
            label_stack.append(label_type)
            tokens.append(summary[j])
            fact_attn_stack.append(attn_dists[j])
        elif label_type == "end":
            pop_label = label_stack.pop()
            if pop_label == "fact":
                tokens.append(summary[j])
        elif label_type == "token":
            tokens.append(summary[j])
        elif label_type == "reference":
            continue
        else:
            label_stack.append(label_type)
    return tokens, fact_attn_stack

def json_out(sentence_list, summary_tokens, attn_dists):
    context = "\t".join(sentence_list[:-1])
    summary = " ".join(summary_tokens)
    json_dict = {}
    json_dict["context"] = context
    json_dict["summary"] = summary
    json_dict["attn_dists"] = attn_dists
    return json.dumps(json_dict)

def highlight_score(label):
    stop_words = [term.strip() for term in open("stop_words.txt")]
    glove_model = Glove.load('glove.model')
    fpout = open('tmp_output/' + label + '.json', 'w')
    for i, line in enumerate(open("tmp_data/corpus_g2g_" + label + ".txt")):
        sentence_list = line.strip().split("\t")
        flist = [item.split(' ') for item in sentence_list]
        context, article, article_index = flatten_vec(flist[0:-1])
        summary = flist[-1]
        context_emb, context_emb_weight = glove_one_sentence(context, stop_words, glove_model)
        summary_emb, summary_emb_weight = glove_one_sentence(summary, stop_words, glove_model)
        attn_dists = get_attn_dists(context_emb, \
                summary_emb, \
                context_emb_weight, \
                summary_emb_weight, \
                context, \
                summary)
        tokens, alignments = fact_level_attn(attn_dists, context, summary)
        json_str = json_out(sentence_list, tokens, alignments)
        fpout.write(json_str + "\n")

def highlight_score_split(label):
    for filename in os.listdir('tmp_data'):
        filename = filename.split('.')[0]
        flist = filename.split('_')
        if flist[2] == label:
            flabel = flist[2] + '_' + flist[3]
            os.system("nohup python glove_alignment.py highlight_score " + flabel + " &")

if __name__ == '__main__':
    if sys.argv[1] == 'data':
        read_data(sys.argv[2])
    if sys.argv[1] == 'split':
        split_data(sys.argv[2])
    if sys.argv[1] == 'highlight_score':
        highlight_score(sys.argv[2])
    if sys.argv[1] == 'highlight_score_split':
        highlight_score_split(sys.argv[2])
