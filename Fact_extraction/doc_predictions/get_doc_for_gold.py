#coding=utf8

import sys, os, json

def version_1():
    srcs = [line.strip().lower() for line in open("./original_data/xsum_test_src.jsonl")]
    tgts = [line.strip().lower() for line in open("./original_data/xsum_test_tgt.jsonl")]
    pairs = {}
    for i, tgt in enumerate(tgts):
        pairs[tgt] = srcs[i]

    gold_path = "xsum-model-predictions/gold-test-sentences.txt"
    if sys.argv[1] == "tconvs2s":
        pred_path = "xsum-model-predictions/topic-convs2s-test-output.txt"
    if sys.argv[1] == "ptgen":
        pred_path = "xsum-model-predictions/ptgen-test-output.txt"

    golds = [line.strip().lower() for line in open(gold_path)]
    preds = [line.strip().lower() for line in open(pred_path)]

    fpout_src = open(sys.argv[1] + "_pred.src", "w")
    fpout_tgt = open(sys.argv[1] + "_pred.tgt", "w")
    for i, gold in enumerate(golds):
        if gold not in pairs:
            continue
        doc = pairs[gold]
        pred = preds[i]
        json_obj = {"document": doc, "pred": pred}
        fpout_src.write(doc + "\n")
        fpout_tgt.write(pred + "\n")
    fpout_src.close()
    fpout_tgt.close()

def load_summary_2_id(filename):
    sid = {}
    for line in open(filename):
        json_obj = json.loads(line.strip())
        summ = json_obj["summ"].lower()
        doc_id = json_obj["doc_id"]
        sid[summ] = doc_id
    return sid

def lowercase(parse):
    flist = parse.split(' ')
    lower_list = []
    for word in flist:
        if word[0] != "(" and word != ")":
            lower_list.append(word.lower())
        else:
            lower_list.append(word)
    return ' '.join(lower_list)

def load_doc_srl(filename):
    did = {}
    for line in open(filename):
        json_obj = json.loads(line.strip())
        document_trees = json_obj["document_trees"]
        document_trees_raw = [lowercase(item.strip()) for item in document_trees if item.strip() != ""]
        doc_id = json_obj["filename"].replace(".data", "")
        did[doc_id] = "\t".join(document_trees_raw)
    return did

def search_doc_srl(sid, doc_srl, golds, preds):
    n_golds = []
    n_preds = []
    docs = []
    for i, g in enumerate(golds):
        if g not in sid:
            continue
        doc_id = sid[g]
        if doc_id not in doc_srl:
            continue
        srl = doc_srl[doc_id]
        n_golds.append(g)
        n_preds.append(preds[i])
        docs.append(srl)
    return n_golds, n_preds, docs


def get_doc_srl(pred_path):
    sid = load_summary_2_id("original_data/summary_2_id.txt")
    doc_srl = load_doc_srl("original_data/xsum_test.jsonl")

    gold_path = "xsum-model-predictions/gold-test-sentences.txt"
    golds = [line.strip().lower() for line in open(gold_path)]
    preds = [line.strip().lower() for line in open(pred_path)]

    n_golds, n_preds, doc_srl = search_doc_srl(sid, doc_srl, golds, preds)

    fpout_src = open(sys.argv[1] + "_pred.src", "w")
    fpout_tgt = open(sys.argv[1] + "_pred.tgt", "w")
    fpout_gold = open(sys.argv[1] + "_pred.gold", "w")

    for item in doc_srl:
        fpout_src.write(item + "\n")
    for item in n_golds:
        fpout_gold.write(item + "\n")
    for item in n_preds:
        fpout_tgt.write(item + "\n")

    fpout_tgt.close()
    fpout_gold.close()
    fpout_src.close()

if __name__ == '__main__':
    if sys.argv[1] == "tconvs2s":
        pred_path = "xsum-model-predictions/topic-convs2s-test-output.txt"
    if sys.argv[1] == "ptgen":
        pred_path = "xsum-model-predictions/ptgen-test-output.txt"
    get_doc_srl(pred_path)
