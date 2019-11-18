#coding=utf8

import sys
import os
import json
sys.path.append(os.path.abspath('../Document_highlight.BERT/'))
from highlight import *
from highlight_alignment import _one_file, preprocess

if __name__ == '__main__':
    if sys.argv[1] == "highlight":
        src_path = "test_50.src"
        tgt_path = "test_50.tgt"
        fpout_path = "./test_50.hl"
        dataset = DataSet(src_path, tgt_path, fpout_path)
        dataset.preprocess_mult()
    if sys.argv[1] == "token_weight":
        hl_file = "./test_50.hl"
        json_file = "test_50.json"
        json_obj = _one_file(hl_file, json_file)
    if sys.argv[1] == "preprocess":
        data_list = preprocess("test_50.json")
        fpout = open("../Document_highlight.BERT/highlights.bert/s2s_alignment/xsum_test_src.jsonl", "w")
        for obj in data_list:
            fpout.write(json.dumps(obj) + "\n")
        fpout.close()
        # Fake data
        fpout = open("../Document_highlight.BERT/highlights.bert/s2s_alignment/xsum_train_src.jsonl", "w")
        for obj in data_list:
            fpout.write(json.dumps(obj) + "\n")
        fpout.close()
        # Fake data
        fpout = open("../Document_highlight.BERT/highlights.bert/s2s_alignment/xsum_dev_src.jsonl", "w")
        for obj in data_list:
            fpout.write(json.dumps(obj) + "\n")
        fpout.close()
