#coding=utf8

import sys, os
import json
import numpy as np

def get_metrics(attn_dists):
    tok_score = []
    for sentence in attn_dists:
        for item in sentence:
            tok_score.extend(item)
    tok_score = np.array(tok_score)
    max_tok = max(tok_score)
    min_tok = min(tok_score)
    mean_tok = np.mean(tok_score)
    std_tok = np.std(tok_score)
    return [max_tok, min_tok, mean_tok, std_tok]

if __name__ == '__main__':
    metrics_list = []
    metrics = ["max_tok", "min_tok", "mean_tok", "std_tok"]
    for i in range(3):
        for line in open("/scratch/xxu/highlights/for_alignment/dev_" + str(i) + ".jsonl"):
            json_obj = json.loads(line.strip())
            attn_dists = json_obj["attn_dists"]
            metrics_list.append(get_metrics(attn_dists[1:]))
    metrics_list = np.array(metrics_list)
    ex_num = len(metrics_list)
    for i in range(len(metrics_list[0])):
        print (metrics[i], sum(metrics_list[:, i])/ex_num)
