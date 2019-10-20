#coding=utf8
import os, sys, json
sys.path.append(os.path.abspath('./rouge/rouge'))
from rouge import Rouge, FilesRouge
from hrouge import HRouge, FilesHRouge

def get_docs():
    ids = []
    for filename in os.listdir("../HROUGE_data/documents/"):
        ids.append(filename.split(".")[0])
    fpout_dir = "./50_docs/"
    for filename in os.listdir("raw_data"):
        for line in open("raw_data/" + filename):
            json_obj = json.loads(line.strip())
            if "filename" in json_obj:
                file_id = json_obj["filename"].split(".")[0]
                if file_id in ids:
                    document_tree = " ".join([item.lower() for item in json_obj["document"]])
                    fpout = open(fpout_dir + file_id + ".data", "w")
                    fpout.write(document_tree + "\n")
                    fpout.close()

def get_file_rouge(ref_dir, hyp_dir):
    ref_map = {}; hyp_map = {}
    for filename in os.listdir(ref_dir):
        file_id = filename.split(".")[0]
        with open(ref_dir + filename, 'r') as file:
            ref_map[file_id] = file.read().strip()
    for filename in os.listdir(hyp_dir):
        file_id = filename.split(".")[0]
        with open(hyp_dir + filename, 'r') as file:
            hyp_map[file_id] = file.read().strip()
    fpout_ref = open("tmp.ref", "w")
    fpout_hyp = open("tmp.hyp", "w")
    for file_id in ref_map:
        fpout_ref.write(ref_map[file_id] + "\n")
        fpout_hyp.write(hyp_map[file_id] + "\n")
    fpout_ref.close()
    fpout_hyp.close()

    # get rouge
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores("tmp.hyp", "tmp.ref", avg=True)
    for item in scores:
        print (item, scores[item])

def get_file_hrouge(ref_tokens, ref_token_weights, hyp_dir):
    hyp_map = {}
    for filename in os.listdir(hyp_dir):
        file_id = filename.split(".")[0]
        with open(hyp_dir + filename, 'r') as file:
            hyp_map[file_id] = file.read().strip()

    refs = []; hyps = []; ref_weights = []
    for file_id in ref_tokens:
        refs.append(ref_tokens[file_id])
        ref_weights.append(ref_token_weights[file_id])
        hyps.append(hyp_map[file_id])

    # get rouge
    files_rouge = FilesHRouge()
    scores = files_rouge.get_scores(hyps, refs, ref_weights, avg=True)
    for item in scores:
        print (item, scores[item])

def get_token_weight(ref_dir):
    token_weights = {}
    tokens = {}
    for filename in os.listdir(ref_dir):
        file_id = filename.split(".")[0]
        with open(ref_dir + filename, 'r') as file:
            json_obj = json.loads(file.read().strip())
        tokens[file_id] = " ".join(json_obj['article_lst'])
        token_weights[file_id] = json_obj['attn_dists'][0]
    return tokens, token_weights

if __name__ == '__main__':
    if sys.argv[1] == "get_docs":
        get_docs()
    elif sys.argv[1] == "rouge":
        ref_dir = "./50_docs/"
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_file_rouge(ref_dir, hyp_dir)
    elif sys.argv[1] == "hrouge":
        ref_dir = "./Bert_token_weight/"
        ref_tokens, ref_token_weights = token_weights = get_token_weight(ref_dir)
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_file_hrouge(ref_tokens, ref_token_weights, hyp_dir)
