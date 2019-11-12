#coding=utf8

import sys
import os
import json
import math
import copy
import numpy as np
from scipy.spatial.distance import cosine
import torch
#from transformers import *
from pytorch_transformers import *
import multiprocessing

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

THRESHOLD_FACT = 0.83
THRESHOLD_PHRASE = 0.8
THRESHOLD_TOKEN = 0.85
BATCH_SIZE = 64
TRUNC_SIZE = 512

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

def bert_sentence(line, stop_words, embeddings, truncate_id):
    embs = []
    tokens = []
    embs_weight = []
    type_stack = []
    label_stack = []
    phrase_id_stack = []
    fact_id_stack = []
    term_id = 0
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
            if l_type == "reference":
                embs.append([])
                tokens.append([])
                embs_weight.append([])
                continue
            if (item in stop_words) or i >= truncate_id:
                # deal with reference
                embs.append([])
                tokens.append([])
                embs_weight.append([])
            else:
                bert_emb = embeddings[term_id]
                embs.append([bert_emb])
                embs_weight.append([1])
                tokens.append([item])
                if len(type_stack) == 0:
                    continue
                elif type_stack[-1] == "phrase":
                    if label_stack[-1] not in ['(ARGM-TMP']:
                        embs[phrase_id_stack[-1]].append(bert_emb)
                        embs_weight[phrase_id_stack[-1]].append(1)
                        tokens[phrase_id_stack[-1]].append(item)
                embs[fact_id_stack[-1]].append(bert_emb)
                embs_weight[fact_id_stack[-1]].append(1)
                tokens[fact_id_stack[-1]].append(item)
            term_id += 1
    return embs, embs_weight, tokens

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
    if phrase_num > 0 and matched_phrase / phrase_num > 0.7:
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
    if token_num == 0 or matched_token / token_num < 0.7:
        return "not related"
    return "rephrase"

def highlight_score(ex, stop_words):
    context_emb, context_emb_weight, context_tokens = bert_sentence(ex.ctx, stop_words, ex.ctx_embs, ex.trunc_ctx_id)
    summary_emb, summary_emb_weight, summary_tokens = bert_sentence(ex.sum, stop_words, ex.sum_embs, len(ex.sum)+1)
    attn_dists, p_gens = get_attn_dists(context_emb, \
            summary_emb, \
            context_emb_weight, \
            summary_emb_weight, \
            ex.ctx, \
            ex.sum, \
            context_tokens, summary_tokens)
    class_type = doc_summary_classifier(p_gens, ex.sum, summary_emb)
    json_obj = {}
    json_obj["article_lst"] = ex.ctx
    json_obj["decoded_lst"] = ex.sum
    json_obj["abstract_str"] = class_type
    json_obj["attn_dists"] = [item.tolist() for item in attn_dists]
    json_obj["p_gens"] = p_gens
    json_obj["ctx_trees"] = ex.ctx_trees
    json_obj["sum_tree"] = ex.sum_tree
    return json.dumps(json_obj)

class DocSumPair:
    def __init__(self, ctx_trees, sum_tree, tokenizer):
        # For getting embs
        self.ctx_trees = ctx_trees
        self.sum_tree = sum_tree
        self.cls = "[CLS]"
        self.cls_id = tokenizer.convert_tokens_to_ids([self.cls])[0]

        self.ctx_terms = [self.get_tokens(ctx_tree) for ctx_tree in self.ctx_trees]
        self.sum_terms = self.get_tokens(self.sum_tree)

        self.ctx_tok = [tokenizer.tokenize(line) for line in self.ctx_terms]
        self.sum_tok = tokenizer.tokenize(self.sum_terms)

        self.add_cls()
        self.truncate_id = self.get_truncate_id()

        self.ctx_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in self.ctx_tok]
        self.sum_ids = tokenizer.convert_tokens_to_ids(self.sum_tok)

        self.for_bert = self.preprocess()

        # For getting alignments
        self.ctx, self.trunc_ctx_id = self.get_ctx()
        self.sum = self.sum_tree.split(" ")

        self.trunc_ctx_terms = self.get_trunc_terms()
        self.trunc_sum_terms = self.sum_terms.split(" ")

        self.trunc_ctx_tok = self.get_trunc_tok()
        self.trunc_sum_tok = self.sum_tok

    def preprocess(self):
        ids = self.sum_ids
        for i in range(self.truncate_id):
            ids.extend(self.ctx_ids[i])
        return ids

    def add_cls(self):
        for i in range(len(self.ctx_tok)):
            self.ctx_tok[i].append(self.cls)
        self.sum_tok.append(self.cls)

    def get_truncate_id(self):
        index = 0; size = len(self.sum_tok)
        while True:
            if index >= len(self.ctx_tok):
                break
            #self.ctx_tok[index].append(self.cls)
            size += len(self.ctx_tok[index])
            if size >= TRUNC_SIZE:
                break
            index += 1
        return index

    def get_trunc_tok(self):
        toks = []
        for i in range(self.truncate_id):
            toks.extend(self.ctx_tok[i])
        return toks

    def get_trunc_terms(self):
        toks = []
        for i in range(self.truncate_id):
            toks.extend(self.ctx_terms[i].split(" "))
        return toks

    def get_ctx(self):
        trunc_ctx_id = 0
        ctx = []
        for i in range(len(self.ctx_trees)):
            toks = self.ctx_trees[i].split(" ")
            ctx.extend(toks)
            if i < self.truncate_id:
                trunc_ctx_id += len(toks)
        return ctx, trunc_ctx_id

    def get_tokens(self, tree):
        tokens = []
        for item in tree.split(" "):
            if len(item) == 0:
                continue
            l_type = label_classify(item)
            if l_type in ["token"]:
                tokens.append(item)
        return " ".join(tokens)

    def merge_tokens(self, tok, terms, emb):
        index = 0; 
        new_emb = []; new_tok = []
        while index < len(tok):
            if tok[index].startswith("##"):
                new_emb[-1] += emb[index]
                new_tok[-1] += tok[index][2:]
            elif tok[index] != "[CLS]":
                if len(new_tok) > 0 and new_tok[-1] != terms[len(new_tok)-1]:
                    new_emb[-1] += emb[index]
                    new_tok[-1] += tok[index]
                else:
                    new_emb.append(emb[index])
                    new_tok.append(tok[index])
            index += 1
        return new_tok, new_emb

    def get_emb(self, emb):
        sum_emb = emb[:len(self.sum_tok), :]
        ctx_emb = emb[len(self.sum_tok):len(self.for_bert), :]
        self.recovered_sum_terms, self.sum_embs = self.merge_tokens(self.trunc_sum_tok, self.trunc_sum_terms, sum_emb)
        self.recovered_ctx_terms, self.ctx_embs = self.merge_tokens(self.trunc_ctx_tok, self.trunc_ctx_terms, ctx_emb)

    def write_out_emb(self, fpout):
        json_obj = {}
        json_obj["ctx"] = self.ctx
        json_obj["sum"] = self.sum
        json_obj["ctx_trees"] = self.ctx_trees
        json_obj["sum_tree"] = self.sum_tree
        json_obj["ctx_embs"] = [item.tolist() for item in self.ctx_embs]
        json_obj["sum_embs"] = [item.tolist() for item in self.sum_embs]
        json_obj["trunc_ctx_id"] = self.trunc_ctx_id
        json_obj["trunc_sum_id"] = len(self.sum)+1
        fpout.write(json.dumps(json_obj) + "\n")

class DataSet:
    def __init__(self, src_path, tgt_path, fpout_path, thred_num=10):
        self.thred_num = thred_num
        self.model_class = BertModel
        self.tokenizer_class = BertTokenizer
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = self.model_class.from_pretrained(self.pretrained_weights).to('cuda')
        self.pool = multiprocessing.Pool(processes=self.thred_num)

        self.src_path = src_path
        self.tgt_path = tgt_path
        self.in_src = [line.strip().encode("ascii", "ignore").decode("ascii", "ignore") for line in open(self.src_path)]
        self.in_tgt = [line.strip().encode("ascii", "ignore").decode("ascii", "ignore") for line in open(self.tgt_path)]

        self.stop_words = [term.strip() for term in open(CUR_DIR + "/stop_words.txt")]

        self.fpout = open(fpout_path, "w")

    def clean(self, line):
        flist = [item for item in line.strip().split(" ") if len(item) > 0]
        new_tok = []; new_type = []
        for item in flist:
            l_type = label_classify(item)
            if l_type == "end" and new_type[-1] in ["fact", "phrase"]:
                new_tok.pop(); new_type.pop()
                continue
            else:
                new_tok.append(item)
                new_type.append(l_type)
        return " ".join(new_tok)

    def pad(self, examples):
        batch_id = [copy.deepcopy(ex.for_bert) for ex in examples]
        max_len = self.get_max_len(batch_id)
        for i in range(len(batch_id)):
            pad_list = [0] * (max_len-len(batch_id[i]))
            batch_id[i].extend(pad_list)
        return batch_id

    def get_max_len(self, batch_id):
        return max([len(item) for item in batch_id])

    def preprocess(self):
        tmp_examples = []
        for i, src in enumerate(self.in_src):
            src_list = []
            for item in src.split("\t"):
                clean_item = self.clean(item)
                if clean_item != "":
                    src_list.append(clean_item)
            tgt = self.clean(self.in_tgt[i])
            tmp_examples.append(DocSumPair(src_list, tgt, self.tokenizer))
            if (i+1)%BATCH_SIZE == 0:
                batch_id = self.pad(tmp_examples)
                batch_id = torch.tensor(batch_id).to('cuda')
                with torch.no_grad():
                    last_hidden_states = self.model(batch_id)[0]
                for i, ex in enumerate(tmp_examples):
                    ex.get_emb(last_hidden_states[i])
                    json_str = highlight_score(ex, self.stop_words)
                    self.fpout.write(json_str + "\n")
                del tmp_examples[:]
        if len(tmp_examples) > 0:
            batch_id = self.pad(tmp_examples)
            batch_id = torch.tensor(batch_id).to('cuda')
            with torch.no_grad():
                last_hidden_states = self.model(batch_id)[0]
            for i, ex in enumerate(tmp_examples):
                ex.get_emb(last_hidden_states[i])
                json_str = highlight_score(ex, self.stop_words)
                self.fpout.write(json_str + "\n")

    def preprocess_mult(self):
        tmp_examples = []
        for i, src in enumerate(self.in_src):
            src_list = []
            for item in src.split("\t"):
                clean_item = self.clean(item)
                if clean_item != "":
                    src_list.append(clean_item)
            tgt = self.clean(self.in_tgt[i])
            tmp_examples.append(DocSumPair(src_list, tgt, self.tokenizer))
            if (i+1)%BATCH_SIZE == 0:
                batch_id = self.pad(tmp_examples)
                batch_id = torch.tensor(batch_id).to('cuda')
                with torch.no_grad():
                    last_hidden_states = self.model(batch_id)[0].cpu().numpy()

                # multi threads
                exs = []
                for j, ex in enumerate(tmp_examples):
                    exs.append((ex, last_hidden_states[j], self.stop_words))
                result_list = self.pool.map(multiprocessing_func, exs)
                for js in result_list:
                    self.fpout.write(js + "\n")
                del tmp_examples[:]

        if len(tmp_examples) > 0:
            batch_id = self.pad(tmp_examples)
            batch_id = torch.tensor(batch_id).to('cuda')
            with torch.no_grad():
                last_hidden_states = self.model(batch_id)[0].cpu().numpy()
            # multi threads
            exs = []
            for j, ex in enumerate(tmp_examples):
                exs.append((ex, last_hidden_states[j], self.stop_words))
            result_list = self.pool.map(multiprocessing_func, exs)
            for js in result_list:
                if js != "":
                    self.fpout.write(js + "\n")

    def get_BERT_emb(self):
        tmp_examples = []
        for i, src in enumerate(self.in_src):
            src_list = []
            for item in src.split("\t"):
                clean_item = self.clean(item)
                if clean_item != "":
                    src_list.append(clean_item)
            tgt = self.clean(self.in_tgt[i])
            tmp_examples.append(DocSumPair(src_list, tgt, self.tokenizer))
            if (i+1)%BATCH_SIZE == 0:
                batch_id = self.pad(tmp_examples)
                batch_id = torch.tensor(batch_id).to('cuda')
                with torch.no_grad():
                    last_hidden_states = self.model(batch_id)[0].cpu().numpy()
                for j, ex in enumerate(tmp_examples):
                    ex.get_emb(last_hidden_states[j])
                    ex.write_out_emb(self.fpout_emb)

def multiprocessing_func(args):
    (ex, last_hidden_states, stop_words) = args
    try:
        ex.get_emb(last_hidden_states)
        json_str = highlight_score(ex, stop_words)
    except:
        return ""
    return json_str

if __name__ == '__main__':
    if sys.argv[1] == "data":
        thred_num = 20
        label = sys.argv[2]
        src_path = "./tmp_data/corpus_g2g_" + label + "_src_.txt"
        tgt_path = "./tmp_data/corpus_g2g_" + label + "_tgt_.txt"
        fpout_path = './highlights.bert/xsum_' + label + ".jsonl"
        dataset = DataSet(src_path, tgt_path, fpout_path, label, thred_num)
        dataset.preprocess_mult()
    if sys.argv[1] == "test":
        thred_num = 1
        src_path = "./test_data/test_src.txt"
        tgt_path = "./test_data/test_tgt.txt"
        srcs = [line.strip() for line in open(src_path)]
        tgts = [line.strip() for line in open(tgt_path)]
        for i, src in enumerate(srcs):
            fpout_tmp = open("test_data/tmp.src", "w")
            fpout_tmp.write(src + "\n")
            fpout_tmp.close()
            fpout_tmp = open("test_data/tmp.tgt", "w")
            fpout_tmp.write(tgts[i] + "\n")
            fpout_tmp.close()
            src_path = "test_data/tmp.src"
            tgt_path = "test_data/tmp.tgt"
            fpout_path = './test_data/' + str(i) + ".jsonl"
            dataset = DataSet(src_path, tgt_path, fpout_path, thred_num)
            dataset.preprocess_mult()
