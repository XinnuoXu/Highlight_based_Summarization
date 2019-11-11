#coding=utf8

import sys
import os
sys.path.append(os.path.abspath('../Document_highlight.BERT/'))
from highlight import *
from highlight_HROUGED import *

if __name__ == '__main__':
    ids = []
    for filename in os.listdir("../HROUGE_data/documents/"):
        ids.append(filename.split(".")[0])
    if sys.argv[1] == "highligh":
        for file_id in ids:
            src_path = "./50_trees/" + file_id + ".src"
            tgt_path = "./50_trees/" + file_id + ".tgt"
            fpout_path = "./Bert_highlight/" + file_id + ".hl"
            dataset = DataSet(src_path, tgt_path, fpout_path)
            dataset.preprocess()
    if sys.argv[1] == "token_weight":
        for file_id in ids:
            file_selected_articles = "./Bert_highlight/" + file_id + ".hl"
            with open(file_selected_articles, 'r') as file:
                line = file.read().strip()
            json_obj = highlight(line)
            fpout = open("./Bert_token_weight/" + file_id + ".json", "w")
            fpout.write(json.dumps(json_obj) + "\n")
            fpout.close()

