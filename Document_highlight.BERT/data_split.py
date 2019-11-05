#coding=utf8

import sys
import os

def split_data(label):
    contexts = []; summaries = []
    for line in open("../../fact_data/g2g/xsum_" + label + "_src.jsonl"):
        contexts.append(line.strip())
    for line in open("../../fact_data/g2g/xsum_" + label + "_tgt.jsonl"):
        summaries.append(line.strip())
    fpout_src = open("tmp.src", "w")
    fpout_tgt = open("tmp.tgt", "w")
    for i in range(len(summaries)):
        if i % 50829 == 0:
            fid = str(int(i / 50829))
            fpout_src.close()
            fpout_tgt.close()
            fpout_src = open("tmp_data/corpus_g2g_" + label + "_" + fid + "_src_.txt", "w")
            fpout_tgt = open("tmp_data/corpus_g2g_" + label + "_" + fid + "_tgt_.txt", "w")
        summary = summaries[i]
        context = contexts[i]
        fpout_src.write(context + "\n")
        fpout_tgt.write(summary + "\n")
    fpout_src.close()
    fpout_tgt.close()
    os.system("rm tmp.src")
    os.system("rm tmp.tgt")
    
if __name__ == '__main__':
    split_data(sys.argv[1])
