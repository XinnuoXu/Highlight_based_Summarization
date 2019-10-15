#coding=utf8

import sys, os
if __name__ == '__main__':
    label = sys.argv[1]
    file_num = int(sys.argv[2])
    fpout_src = open("final_output/xsum_" + label + "_src.jsonl", "w")
    fpout_tgt = open("final_output/xsum_" + label + "_tgt.jsonl", "w")
    for i in range(file_num+1):
        filename_src = "tmp_output/xsum_" + label + "_" + str(i) + "_src.jsonl"
        filename_tgt = "tmp_output/xsum_" + label + "_" + str(i) + "_tgt.jsonl"
        for line in open(filename_src):
            fpout_src.write(line.strip() + "\n")
        for line in open(filename_tgt):
            fpout_tgt.write(line.strip() + "\n")
    fpout_src.close()
    fpout_tgt.close()

