#coding=utf8

import sys
import os
import json

def line_lower(line):
    return " ".join([item if item[0] == "(" else item.lower() for item in line.split(" ")])

if __name__ == '__main__':
    ids = []
    for filename in os.listdir("../HROUGE_data/documents/"):
        ids.append(filename.split(".")[0])

    for filename in os.listdir("raw_data"):
        for line in open("raw_data/" + filename):
            json_obj = json.loads(line.strip())
            if "filename" in json_obj:
                file_id = json_obj["filename"].split(".")[0]
            if file_id in ids:
                fpout_src = open("./50_trees/" + file_id + ".src", "w")
                fpout_tgt = open("./50_trees/" + file_id + ".tgt", "w")
                summary_tree = line_lower(json_obj["summary_tree"])
                document_tree = "\t".join([line_lower(item) for item in json_obj["document_trees"]])
                fpout_src.write(document_tree + "\n")
                fpout_tgt.write(summary_tree + "\n")
                fpout_src.close()
                fpout_tgt.close()
