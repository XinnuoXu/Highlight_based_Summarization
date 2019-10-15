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

THRESHOLD_FACT = 0.7
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
    tokens = []
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
            tokens.append([])
            embs_weight.append([])
        elif l_type == "end":
            embs.append([])
            tokens.append([])
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
                tokens.append([])
                embs_weight.append([])
            else:
                glv_emb = glove_model.word_vectors[glove_model.dictionary[item]]
                embs.append([glv_emb])
                embs_weight.append([1])
                tokens.append([item])
                if len(type_stack) == 0:
                    continue
                elif type_stack[-1] == "phrase":
                    if label_stack[-1] not in ['(ARGM-TMP']:
                        embs[phrase_id_stack[-1]].append(glv_emb)
                        embs_weight[phrase_id_stack[-1]].append(1)
                        tokens[phrase_id_stack[-1]].append(item)
                embs[fact_id_stack[-1]].append(glv_emb)
                embs_weight[fact_id_stack[-1]].append(1)
                tokens[fact_id_stack[-1]].append(item)
                #for f_id in fact_id_stack[:-1]:
                #    embs[f_id].append(glv_emb)
                #    embs_weight[f_id].append(0.3)
                #    tokens[f_id].append(item)
    return embs, embs_weight, tokens

def flatten_vec(vec):
    res = []
    article = []
    article_index = []
    for i, llist in enumerate(vec):
        article.append(" ".join(llist))
        article_index.extend([i]*len(llist))
        res.extend(llist)
    return res, article, article_index

def set_2_set_sum(s_emb, c_emb, s_emb_w, c_emb_w):
    attn_score = 0.0
    if len(c_emb) == 0:
        return attn_score
    c_emb = sum(c_emb) / len(c_emb)
    s_emb = sum(s_emb) / len(s_emb)
    sim_score = max(0, 1-cosine(s_emb, c_emb))
    return sim_score

def set_2_set(s_emb, c_emb, s_emb_w, c_emb_w, func="max"):
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
                #sim_score = sim_score if sim_score > 0.6 else 0
                scores.append(sim_score * s_weight * c_weight)
        if func == "max":
            term_score = max(scores)
        elif func == "sum":
            term_score = sum(scores)
        attn_score += term_score
    if non_stop_word_num == 0:
        return 0
    else:
        return attn_score / non_stop_word_num

def get_attn_dists(context_embs, summary_emb, context_emb_weight, summary_emb_weight, context, summary, context_tokens, summary_tokens):
    attn_dists = []
    p_gens = []
    term_len = len(context_embs)
    for i, s_emb in enumerate(summary_emb):
        if len(s_emb) == 0:
            attn_dists.append(np.zeros(term_len))
            p_gens.append(-1)
            continue
        attn = []
        for j, c_emb in enumerate(context_embs):
            if label_classify(context[j]) != label_classify(summary[i]):
                attn.append(0)
            else:
                score = set_2_set(s_emb, c_emb, summary_emb_weight[i], context_emb_weight[j], func="max")
                attn.append(score)
        sum_label = label_classify(summary[i])
        attn_dists.append(np.array(attn))
        p_gens.append(max(attn))

        '''
        log = []
        for j, item in enumerate(attn):
            log.append(str(attn[j]) + "|" + context[j])
        arr = np.array(attn)
        print (" ".join(log))
        threshold_idx = arr.argsort()[-1:]
        print (summary[i])
        print (summary_tokens[i])
        print ("++++++++++++")
        for idx in threshold_idx:
            print (context[idx])
            print (context_tokens[idx])
            print (arr[idx])
            print ("**********")
        '''
    return attn_dists, p_gens

def doc_summary_classifier(p_gens, summary, summary_emb):
    # fact2fact matching
    fact_scores = []
    fact_match = True
    for i, item in enumerate(summary):
        if label_classify(item) == "fact":
            if len(summary_emb[i]) == 0:
                # fact with only stop words
                continue
            fact_scores.append(p_gens[i])
            if p_gens[i] < THRESHOLD_FACT:
                fact_match = False
                break
    #print ("Fact scores:", fact_scores)
    if fact_match:
        return "fact_match"

    # phrase level matching
    phrase_num = 0.0
    matched_phrase = 0.0
    phrase_scores = []
    for i, item in enumerate(summary):
        if label_classify(item) == "phrase":
            if len(summary_emb[i]) == 0:
                # phrase with only stop words
                continue
            phrase_num += 1
            phrase_scores.append(p_gens[i])
            if p_gens[i] < THRESHOLD_PHRASE:
                continue
            else:
                matched_phrase += 1
    #print ("Phrase scores:", phrase_scores)
    #print (phrase_num, matched_phrase)
    if phrase_num > 0 and matched_phrase / phrase_num > 0.6:
        return "phrase_match"

    # not related
    token_num = 0.0
    matched_token = 0.0
    token_scores = []
    for i, item in enumerate(summary):
        if label_classify(item) == "token":
            token_scores.append(p_gens[i])
            token_num += 1
            if p_gens[i] > -1 and p_gens[i] < THRESHOLD_TOKEN:
                continue
            else:
                matched_token += 1
    #print ("Token scores:", token_scores)
    if token_num == 0 or matched_token / token_num < 0.6:
        return "not related"
    return "rephrase"


def only_tokens(tokens):
    return [tok for tok in tokens if label_classify(tok) in ['token']]

def highlight_score(label):
    stop_words = [term.strip() for term in open("stop_words.txt")]
    glove_model = Glove.load('glove.model')
    fpout = open('/scratch/xxu/highlights/xsum_' + label + '.jsonl', 'w')
    for i, line in enumerate(open("tmp_data/corpus_g2g_" + label + ".txt")):
        sentence_list = line.strip().split("\t")
        flist = [item.split(' ') for item in sentence_list]
        context, article, article_index = flatten_vec(flist[0:-1])
        summary = flist[-1]
        context_emb, context_emb_weight, context_tokens = glove_one_sentence(context, stop_words, glove_model)
        summary_emb, summary_emb_weight, summary_tokens = glove_one_sentence(summary, stop_words, glove_model)
        attn_dists, p_gens = get_attn_dists(context_emb, \
                summary_emb, \
                context_emb_weight, \
                summary_emb_weight, \
                context, \
                summary, \
                context_tokens, summary_tokens)
        class_type = doc_summary_classifier(p_gens, summary, summary_emb)
        #context = only_tokens(context)
        #summary = only_tokens(summary)
        json_obj = {}
        json_obj["article_lst"] = context
        json_obj["decoded_lst"] = summary
        json_obj["abstract_str"] = class_type
        json_obj["attn_dists"] = [item.tolist() for item in attn_dists]
        json_obj["p_gens"] = p_gens
        fpout.write(json.dumps(json_obj) + "\n")
    fpout.close()

def highlight_score_split(label):
    for filename in os.listdir('tmp_data'):
        filename = filename.split('.')[0]
        flist = filename.split('_')
        if flist[2] == label:
            flabel = flist[2] + '_' + flist[3]
            os.system("nohup python quality_classifier.py highlight_score " + flabel + " &")

if __name__ == '__main__':
    if sys.argv[1] == 'data':
        read_data(sys.argv[2])
    if sys.argv[1] == 'split':
        split_data(sys.argv[2])
    if sys.argv[1] == 'highlight_score':
        highlight_score(sys.argv[2])
    if sys.argv[1] == 'highlight_score_split':
        highlight_score_split(sys.argv[2])
