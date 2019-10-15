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

THRESHOLD_FACT = 0.6
THRESHOLD_PHRASE = 0.7
THRESHOLD_TOKEN = 0.8

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
    non_stop_word_num = 0
    for i, s_term in enumerate(s_emb):
        if sum(s_term) == 0:
            continue
        non_stop_word_num += 1
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
    if non_stop_word_num == 0:
        return 0
    else:
        return attn_score / non_stop_word_num

def set_2_set_v1(s_emb, c_emb, s_emb_w, c_emb_w):
    attn_score = 0.0
    if len(c_emb) == 0:
        return attn_score
    for i, s_term in enumerate(s_emb):
        if sum(s_term) == 0:
            continue
        s_weight = s_emb_w[i]
        scores = [0]
        for j, c_term in enumerate(c_emb):
            if sum(c_term) == 0:
                scores.append(0)
            else:
                scores.append(max(0, 1-cosine(s_term, c_term)))
        term_score = max(scores)
        #term_score = term_score if term_score > 0.5 else 0
        term_score = term_score if term_score > 0.7 else 0
        attn_score += term_score
    #return (attn_score / len(s_emb)) / math.log(len(c_emb) + 1, 2)
    return attn_score / len(s_emb)

def is_label(item):
    if item[0] == '(':
        return True
    return False

def get_attn_dists(context_embs, summary_emb, context_emb_weight, summary_emb_weight, context, summary, top_n):
    attn_dists = []
    p_gens = []
    term_len = len(context_embs)
    for i, s_emb in enumerate(summary_emb):
        if len(s_emb) == 0:
            attn_dists.append(np.zeros(term_len).tolist())
            p_gens.append(0)
            continue
        attn = []
        for j, c_emb in enumerate(context_embs):
            if label_classify(context[j]) != label_classify(summary[i]):
                attn.append(0)
            else:
                score = set_2_set(s_emb, c_emb, summary_emb_weight[i], context_emb_weight[j])
                attn.append(score)
        if top_n > 0:
            attn_np = np.array(attn)
            top_n_idx = attn_np.argsort()[-top_n:]
            attn_mask = np.zeros(len(attn))
            for max_id in top_n_idx:
                attn_mask[max_id] = 1
            attn_dists.append(list(attn_mask))
            p_gens.append(max(attn_mask))
        elif top_n == 0:
            attn_max = max(attn)
            if attn_max > 0:
                arr = np.array(attn) / attn_max
            else:
                arr = np.array(attn)
            threshold_idx = arr.argsort()[-3]
            threshold = arr[threshold_idx]
            threshold = max(0.8, threshold)
            attn = [item if item >= threshold else 0 for item in arr]
            attn_dists.append(attn)
            p_gens.append(max(attn))
        else:
            sum_label = label_classify(summary[i])
            attn_dists.append(np.array(attn))
            p_gens.append(max(attn))
    return attn_dists, p_gens

def sentence_selection(attn_dists, context, article, article_index):
    article_set = set()
    for attn_term in attn_dists:
        for i, attn in enumerate(attn_term):
            if attn > 0 and label_classify(context[i]) == "fact":
                article_set.add(article_index[i])
    selected_article = []
    for i, sentence in enumerate(article):
        if i in article_set:
            selected_article.append(article[i])
    return selected_article

def facts_for_doc(attn_dists, context):
    fact_attn = []
    for i, term in enumerate(context):
        label_type = label_classify(term)
        if label_type not in ["fact"]:
            continue
        fact_attn.append(0)
        if label_type == "fact":
            for j, attn_f in enumerate(attn_dists):
                fact_attn[-1] = fact_attn[-1] | (1 if attn_dists[j][i] > 0 else 0)
    return fact_attn

def facts_for_facts(attn_dists, context, summary):
    tmp_attn = []
    for item in attn_dists:
        t_attn = []
        for i, term in enumerate(context):
            if label_classify(term) == "fact":
                t_attn.append(int(item[i]))
        tmp_attn.append(t_attn)
    default_len = len(tmp_attn[0])

    attn_dists = tmp_attn
    fact_attn_stack = []; label_stack = []
    tokens = []; alignments = []
    for j, attn_f in enumerate(attn_dists):
        label_type = label_classify(summary[j])
        if label_type == "fact":
            label_stack.append(label_type)
            fact_attn_stack.append(attn_dists[j])
        elif label_type == "end":
            pop_label = label_stack.pop()
            if pop_label == "fact":
                fact_attn_stack.pop()
        elif label_type == "token":
            if len(fact_attn_stack) == 0:
                zero_list = [0] * default_len
                alignments.append(zero_list)
            else:
                alignments.append(fact_attn_stack[-1])
            tokens.append(summary[j])
        elif label_type == "reference":
            continue
        else:
            label_stack.append(label_type)
    return tokens, alignments

def facts_alignment(attn_dists, context, summary):
    tmp_attn = []
    for item in attn_dists:
        t_attn = []
        for i, term in enumerate(context):
            if label_classify(term) == "fact":
                t_attn.append(int(item[i]))
        tmp_attn.append(t_attn)
    default_len = len(tmp_attn[0])

    attn_dists = tmp_attn
    fact_attn_stack = []; label_stack = []; tokens = []
    for j, attn_f in enumerate(attn_dists):
        label_type = label_classify(summary[j])
        if label_type == "fact":
            tokens.append(summary[j])
            label_stack.append(label_type)
            fact_attn_stack.append(attn_dists[j])
        elif label_type == "end":
            pop_label = label_stack.pop()
            if pop_label == "fact":
                #fact_attn_stack.pop()
                tokens.append(summary[j])
        elif label_type == "token":
            tokens.append(summary[j])
        elif label_type == "reference":
            continue
        else:
            label_stack.append(label_type)
    return tokens, fact_attn_stack

def highlight_score(label):
    stop_words = [term.strip() for term in open("stop_words.txt")]
    glove_model = Glove.load('glove.model')
    fpout_src = open('tmp_output.thres/xsum_' + label + '_src.jsonl', 'w')
    fpout_tgt = open('tmp_output.thres/xsum_' + label + '_tgt.jsonl', 'w')
    for i, line in enumerate(open("tmp_data/corpus_g2g_" + label + ".txt")):
        sentence_list = line.strip().split("\t")
        flist = [item.split(' ') for item in sentence_list]
        context, article, article_index = flatten_vec(flist[0:-1])
        summary = flist[-1]
        context_emb, context_emb_weight = glove_one_sentence(context, stop_words, glove_model)
        summary_emb, summary_emb_weight = glove_one_sentence(summary, stop_words, glove_model)
        attn_dists, p_gens = get_attn_dists(context_emb, \
                summary_emb, \
                context_emb_weight, \
                summary_emb_weight, \
                context, \
                summary, \
                top_n=0)
        #fact_attn = facts_for_doc(attn_dists, context)
        tokens, alignments = facts_for_facts(attn_dists, context, summary)

        '''
        fpout_src.write("\t".join(sentence_list[:-1]) + "\n")
        tgt_line = []
        for j, token in enumerate(tokens):
            tgt_line.append(token)
            ali_str = [str(a) for a in alignments[j]]
            tgt_line.append("".join(ali_str))
        fpout_tgt.write(" ".join(tgt_line) + "\n")
        '''

        '''
        selected_article = sentence_selection(attn_dists, context, article, article_index)
        output_json = {}
        output_json['article_lst'] = context
        output_json['decoded_lst'] = summary
        output_json['abstract_str'] = " ".join(selected_article)
        output_json['attn_dists'] = attn_dists
        output_json['p_gens'] = p_gens
        fpout = open("output/highlight_" + str(i) + ".json", "w")
        fpout.write(json.dumps(output_json) + "\n")
        fpout.close()
        '''

        tokens, fact_attn_stack = facts_alignment(attn_dists, context, summary)
        fact_attn_str = [str(f) for f in fact_attn_stack]
        fpout_tgt.write(" ".join(tokens) + "\t" + " ".join(fact_attn_str) + "\n")
        fpout_src.write("\t".join(sentence_list[:-1]) + "\n")

def highlight_score_split(label):
    for filename in os.listdir('tmp_data'):
        filename = filename.split('.')[0]
        flist = filename.split('_')
        if flist[2] == label:
            flabel = flist[2] + '_' + flist[3]
            os.system("nohup python highlight.py highlight_score " + flabel + " &")

if __name__ == '__main__':
    if sys.argv[1] == 'data':
        read_data(sys.argv[2])
    if sys.argv[1] == 'split':
        split_data(sys.argv[2])
    if sys.argv[1] == 'highlight_score':
        highlight_score(sys.argv[2])
    if sys.argv[1] == 'highlight_score_split':
        highlight_score_split(sys.argv[2])
