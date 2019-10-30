#coding=utf8

import sys
import os
import json

_blacklist = ["these are external links and will open in a new window",
        "share this with",
        "email",
        "facebook",
        "messenger",
        "twitter",
        "pinterest",
        "whatsapp",
        "linkedin",
        "copy this link"]

def _label_classify(item):
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

def _tree_exam(tokens):
    at_at_num = 0
    for i in range(len(tokens)):
        if tokens[i] == "@@":
            at_at_num += 1
            if at_at_num % 2 == 1:
                tokens[i] = "("
            else:
                tokens[i] = ")"
    return tokens

def _trees_exam(trees):
    new_trees = []
    for tree in trees.split('\t'):
        sent = []
        for item in tree.split():
            if _label_classify(item) == "token":
                sent.append(item)
        sent = " ".join(_tree_exam(sent)) + " ."
        new_trees.append(sent)
    return " ".join(new_trees)

def _tree_preprocess(line):
    return " ".join([item if _label_classify(item) != "token" else item.lower() for item in line.split(" ")])

def trees_preprocess(trees, sents):
    new_trees = []
    for i, tree in enumerate(trees):
        sentence = sents[i].lower()
        if sentence in _blacklist:
            continue
        new_trees.append(_tree_preprocess(tree))
    return "\t".join(new_trees)

if __name__ == '__main__':
    ids = []
    for filename in os.listdir("../HROUGE_data/documents/"):
        ids.append(filename.split(".")[0])
    if sys.argv[1] == "data_exam":
        standar_docs = {}; standar_summs = {}
        for line in open("Hardy_HROUGE/highres/debug.data"):
            line = line.strip()
            json_obj = json.loads(line)
            doc_id = json_obj["doc_id"]
            doc = json_obj["document"]
            summary = json_obj["summary"]
            standar_docs[doc_id] = doc
            standar_summs[doc_id] = summary
        for filename in os.listdir("raw_data"):
            for line in open("raw_data/" + filename):
                json_obj = json.loads(line.strip())
                if "filename" in json_obj:
                    file_id = json_obj["filename"].split(".")[0]
                if file_id not in ids:
                    continue
                summary_tree = _tree_preprocess(json_obj["summary_tree"])
                document_tree = json_obj["document_trees"]
                document = json_obj["document"]
                document_tree = trees_preprocess(document_tree, document)
                my_doc = _trees_exam(document_tree)
                doc = standar_docs[file_id]
                if doc != my_doc:
                    doc = doc.split()
                    my_doc = my_doc.split()
                    for i, item in enumerate(doc):
                        if item != my_doc[i]:
                            print (item, my_doc[i])
                            break
                    print (doc)
                    print (my_doc)
    elif sys.argv[1] == "get_trees": 
        for filename in os.listdir("raw_data"):
            for line in open("raw_data/" + filename):
                json_obj = json.loads(line.strip())
                if "filename" in json_obj:
                    file_id = json_obj["filename"].split(".")[0]
                if file_id in ids:
                    fpout_src = open("./50_trees/" + file_id + ".src", "w")
                    fpout_tgt = open("./50_trees/" + file_id + ".tgt", "w")
                    summary_tree = _tree_preprocess(json_obj["summary_tree"])
                    document_tree = json_obj["document_trees"]
                    document = json_obj["document"]
                    document_tree = trees_preprocess(document_tree, document)
                    fpout_src.write(document_tree + "\n")
                    fpout_tgt.write(summary_tree + "\n")
                    fpout_src.close()
                    fpout_tgt.close()
