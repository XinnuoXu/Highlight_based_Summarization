#coding=utf8

import sys
import random

RAW_PATH="../../../fact_data/s2s_bert/xsum_"

def read_doc_summ(label):
    docs = [line.strip().lower() for line in open(RAW_PATH + label + "_src.jsonl")]
    summs = [line.strip().lower() for line in open(RAW_PATH + label + "_tgt.jsonl")]
    return docs, summs

def get_random_summ(summs):
    random_summs = []
    for i, item in enumerate(summs):
        rand_id = random.randint(0, len(summs)-1)
        while rand_id == i:
            rand_id = random.randint(0, len(summs))
        random_summs.append(item + "\t" + summs[rand_id])
    return random_summs

if __name__ == '__main__':
    label = sys.argv[1]
    docs, summs = read_doc_summ(label)
    random_summs = get_random_summ(summs)
    fpout_s = open("../raw_data/xsum_" + label + "_src.jsonl", 'w')
    fpout_g = open("../raw_data/xsum_" + label + "_tgt.jsonl", 'w')
    fpout_s.write("\n".join(docs))
    fpout_g.write("\n".join(random_summs))
    fpout_s.close()
    fpout_g.close()
