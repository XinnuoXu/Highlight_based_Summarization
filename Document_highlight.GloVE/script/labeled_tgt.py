#coding=utf8

import sys
import os
import json
import math
import numpy as np
from nltk.stem import PorterStemmer
porter = PorterStemmer()

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

def label(summary):
    fact_stack = []; label_stack = []; 
    str_list = []; fact_list = []; token_list = []
    for j, word in enumerate(summary):
        label_type = label_classify(summary[j])
        if label_type == "fact":
            label_stack.append(label_type)
            if len(word.split('-')) == 2:
                stemed = "F-" + porter.stem((word.split('-')[1]))
                fact_stack.append(stemed)
            else:
                fact_stack.append("F0")
            fact_list.append(fact_stack[-1])
            str_list.append("(F" + u"￨" + fact_stack[-1])
            token_list.append("(F")
        elif label_type == "phrase":
            label_stack.append(label_type)
        elif label_type == "end":
            pop_label = label_stack.pop()
            if pop_label == "fact":
                str_list.append(")" + u"￨" + fact_stack[-1])
                fact_list.append(fact_stack[-1])
                token_list.append(")")
                fact_stack.pop()
        elif label_type == "token":
            if len(fact_stack) == 0:
                continue
            fact_list.append(fact_stack[-1])
            str_list.append(summary[j] + u"￨" + fact_stack[-1])
            token_list.append(summary[j])
    return str_list, fact_list, token_list

def unify_fact_list(fact_list):
    pre_fact = ""
    unify = []
    for item in fact_list:
        if item != pre_fact:
            unify.append(item.encode('utf8'))
        pre_fact = item
    return unify

def labeled_tgt_language_model(dataset):
    fact_dict = {}
    fpout_src = open("/scratch/xxu/labeled_tgt/xsum_" + dataset + "_src.jsonl", "w")
    fpout_tgt = open("/scratch/xxu/labeled_tgt/xsum_" + dataset + "_tgt.jsonl", "w")
    fpout_cst = open("/scratch/xxu/labeled_tgt/xsum_" + dataset + "_cst.jsonl", "w")
    for line in open("../fact_data/g2g/xsum_" + dataset + "_tgt.jsonl"):
        str_list, fact_list, token_list = label(line.strip().decode('utf8').split(" "))
        for i in range(1, len(str_list)):
            fpout_src.write(" ".join(str_list[:i]).encode('utf8') + "\n")
            fpout_cst.write(" ".join(unify_fact_list(fact_list[i:])) + "\n")
            fpout_tgt.write(" ".join(str_list[i:]).encode('utf8') + "\n")
        for item in fact_list:
            if item not in fact_dict:
                fact_dict[item] = 1
            else:
                fact_dict[item] += 1
    fpout_src.close()
    fpout_tgt.close()
    fpout_cst.close()

    '''
    if dataset == "train":
        fpout = open("./labeled_tgt/fact_list.txt", 'w')
        for item in fact_dict:
            if fact_dict[item] > 1:
                fpout.write(item.encode('utf8') + "\n")
    fpout.close()
    '''

def labeled_tgt(dataset):
    fpout_tgt = open("/scratch/xxu/labeled_tgt/xsum_" + dataset + "_tgt.jsonl", "w")
    fpout_cst = open("/scratch/xxu/labeled_tgt/xsum_" + dataset + "_cst.jsonl", "w")
    for line in open("../fact_data/g2g/xsum_" + dataset + "_tgt.jsonl"):
        str_list, fact_list, token_list = label(line.strip().decode('utf8').split(" "))
        fpout_cst.write(" ".join(unify_fact_list(fact_list)) + "\n")
        fpout_tgt.write(" ".join(str_list).encode('utf8') + "\n")
    fpout_tgt.close()
    fpout_cst.close()

def labeled_src(dataset):
    fpout_src = open("/scratch/xxu/labeled_tgt/xsum_" + dataset + "_src.jsonl", "w")
    for line in open("../fact_data/g2g/xsum_" + dataset + "_src.jsonl"):
        doc_list = line.strip().split("\t")
        lebeled_list = []
        for line in doc_list:
            str_list, fact_list, token_list = label(line.strip().decode('utf8').split(" "))
            lebeled_list.append(" ".join(str_list).encode('utf8'))
        fpout_src.write(" ".join(lebeled_list) + "\n")
    fpout_src.close()

if __name__ == '__main__':
    #labeled_tgt(sys.argv[1])
    labeled_src(sys.argv[1])
