#coding=utf8

import sys
import os
sys.path.append(os.path.abspath('../Document_highlight.BERT/'))
from highlight import *
from highlight_HROUGED import *

def merge_file(ids, tgt_dir="./50_trees/"):
    tmp_src = open("tmp.src", "w")
    tmp_tgt = open("tmp.tgt", "w")
    for file_id in ids:
        src_path = "./50_trees/" + file_id + ".src"
        with open(src_path, 'r') as file:
            line = file.read().strip()
        tmp_src.write(line + "\n")

        tgt_path = tgt_dir + "/" + file_id + ".tgt"
        with open(tgt_path, 'r') as file:
            line = file.read().strip().replace(" [ UNK ] ", " ") + "\t" + file_id
        tmp_tgt.write(line + "\n")
    tmp_src.close()
    tmp_tgt.close()
    return "tmp.src", "tmp.tgt"

def split_file(tmp_out, fpout_base="./Bert_highlight/"):
    for line in open(tmp_out):
        line = line.strip()
        file_id = json.loads(line)["doc_id"]
        fpout_path = fpout_base + "/" + file_id + ".hl"
        fpout=open(fpout_path, "w")
        fpout.write(line)
        fpout.close()

def simple_format(tgt_file, tag):
    fpout = open("tmp.tgt." + tag, "w")
    for i, line in enumerate(open(tgt_file)):
        fpout.write(line.strip() + "\t" + str(i) + "\n")
    fpout.close()
    return "tmp.tgt." + tag

if __name__ == '__main__':
    ids = []
    for filename in os.listdir("../HROUGE_data/documents/"):
        ids.append(filename.split(".")[0])
    if sys.argv[1] == "highlight_simple_format":
        tmp_src = sys.argv[2]
        tmp_tgt = sys.argv[3]
        tmp_output = sys.argv[4]
        tmp_tgt = simple_format(tmp_tgt, sys.argv[3].split("/")[-1].split(".")[0])
        dataset = DataSet(tmp_src, tmp_tgt, tmp_output, thred_num=20)
        dataset.preprocess_mult()
    if sys.argv[1] == "highlight":
        if len(sys.argv) == 2:
            tmp_src, tmp_tgt = merge_file(ids)
        else:
            tmp_src, tmp_tgt = merge_file(ids, sys.argv[2])
        tmp_output = "tmp.output"
        dataset = DataSet(tmp_src, tmp_tgt, tmp_output, thred_num=20)
        dataset.preprocess_mult()
        if len(sys.argv) == 2:
            split_file(tmp_output)
        else:
            fpout_path = ".".join(sys.argv[2].split(".")[:-1]) + ".alg"
            if not os.path.exists(fpout_path):
                os.system("mkdir " + fpout_path)
            split_file(tmp_output, fpout_path)
    if sys.argv[1] == "merge":
        for file_id in ids:
            file_selected_articles = "./Bert_highlight/" + file_id + ".hl"
            with open(file_selected_articles, 'r') as file:
                line = file.read().strip()
            json_obj = highlight(line)
            fpout = open("./Bert_token_weight/" + file_id + ".json", "w")
            fpout.write(json.dumps(json_obj) + "\n")
            fpout.close()
    if sys.argv[1] == "phrase2phrase":
        for file_id in ids:
            file_selected_articles = "./Bert_highlight/" + file_id + ".hl"
            with open(file_selected_articles, 'r') as file:
                line = file.read().strip()
            json_obj = phrase_to_phrase(line)
            fpout = open("./Bert_phrase/" + file_id + ".json", "w")
            fpout.write(json.dumps(json_obj) + "\n")
            fpout.close()
    if sys.argv[1] == "fact2fact":
        for file_id in ids:
            file_selected_articles = "./Bert_highlight/" + file_id + ".hl"
            with open(file_selected_articles, 'r') as file:
                line = file.read().strip()
            json_obj = fact_to_fact(line)
            fpout = open("./Bert_fact/" + file_id + ".json", "w")
            fpout.write(json.dumps(json_obj) + "\n")
            fpout.close()



