#coding=utf8
import os, sys, json

sys.path.append(os.path.abspath('./rouge/rouge'))
from rouge import Rouge, FilesRouge
from hrouge import HRouge, FilesHRouge

sys.path.append(os.path.abspath('./Hardy_HROUGE'))
from analyse_ngram import R_rec, R_prec
from hardy_rouge import HR_rec, HR_prec

def label_classify(item):
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

def g2g_token_replace(tokens):
    at_at_num = 0
    quote_num = 0
    for i in range(len(tokens)):
        if tokens[i] == "@@":
            at_at_num += 1
            if at_at_num % 2 == 1:
                tokens[i] = "-lrb-"
            else:
                tokens[i] = "-rrb-"
        elif tokens[i] == "£":
            tokens[i] = "#"
        elif tokens[i] == "\"":
            quote_num += 1
            if quote_num % 2 == 1:
                tokens[i] = "``"
            else:
                tokens[i] = "\'\'"
    return tokens

def _rephrase(article_lst):
    tokens = []
    for i, token in enumerate(article_lst):
        token_type = label_classify(token)
        if token_type == "token":
            tokens.append(token.lower())
    tokens = g2g_token_replace(tokens)
    return " ".join(tokens)

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
                    document_tree = _rephrase(" ".join(json_obj["document_trees"]).split(" "))
                    #document_tree = " ".join([item.lower() for item in json_obj["document"]])
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

def get_hardy_hrouge(ref_map, ref_weights, hyp_dir):
    hyp_map = {}
    for filename in os.listdir(hyp_dir):
        file_id = filename.split(".")[0]
        with open(hyp_dir + filename, 'r') as file:
            hyp_map[file_id] = file.read().strip()

    rec_1s = []; prec_1s = []; f_1s = []; rec_2s = []; prec_2s = []; f_2s = []
    for file_id in ref_map:
        doc = ref_map[file_id]
        summ = hyp_map[file_id]
        weights = ref_weights[file_id]
        rec_1 = HR_rec(1, summ, doc, weights)
        prec_1 = HR_prec(1, summ, doc, weights)
        if rec_1 + prec_1 == 0:
            f_1 = 0
        else:
            f_1 = 2 * rec_1 * prec_1 / (rec_1 + prec_1)
        rec_2 = HR_rec(2, summ, doc, weights)
        prec_2 = HR_prec(2, summ, doc, weights)
        if rec_2 + prec_2 == 0:
            f_2 = 0
        else:
            f_2 = 2 * rec_2 * prec_2 / (rec_2 + prec_2)
        rec_1s.append(rec_1)
        prec_1s.append(prec_1)
        f_1s.append(f_1)
        rec_2s.append(rec_2)
        prec_2s.append(prec_2)
        f_2s.append(f_2)
    print (">> Unigram ROUGE-P: {:2.2f}".format(sum(prec_1s)/len(prec_1s) * 100))
    print (">> Unigram ROUGE-R: {:2.2f}".format(sum(rec_1s)/len(rec_1s) * 100))
    print (">> Bigram ROUGE-P: {:2.2f}".format(sum(prec_2s)/len(prec_2s) * 100))
    print (">> Bigram ROUGE-R: {:2.2f}".format(sum(rec_2s)/len(rec_2s) * 100))

def get_hardy_rouge(ref_dir, hyp_dir):
    ref_map = {}; hyp_map = {}
    for filename in os.listdir(ref_dir):
        file_id = filename.split(".")[0]
        with open(ref_dir + filename, 'r') as file:
            ref_map[file_id] = file.read().strip()
    for filename in os.listdir(hyp_dir):
        file_id = filename.split(".")[0]
        with open(hyp_dir + filename, 'r') as file:
            hyp_map[file_id] = file.read().strip()
    rec_1s = []; prec_1s = []; f_1s = []; rec_2s = []; prec_2s = []; f_2s = []
    for file_id in ref_map:
        doc = ref_map[file_id]
        summ = hyp_map[file_id]
        rec_1 = R_rec(1, summ, doc)
        prec_1 = R_prec(1, summ, doc)
        if rec_1 + prec_1 == 0:
            f_1 = 0
        else:
            f_1 = 2 * rec_1 * prec_1 / (rec_1 + prec_1)
        rec_2 = R_rec(2, summ, doc)
        prec_2 = R_prec(2, summ, doc)
        if rec_2 + prec_2 == 0:
            f_2 = 0
        else:
            f_2 = 2 * rec_2 * prec_2 / (rec_2 + prec_2)
        rec_1s.append(rec_1)
        prec_1s.append(prec_1)
        f_1s.append(f_1)
        rec_2s.append(rec_2)
        prec_2s.append(prec_2)
        f_2s.append(f_2)
    print (">> Unigram ROUGE-P: {:2.2f}".format(sum(prec_1s)/len(prec_1s) * 100))
    print (">> Unigram ROUGE-R: {:2.2f}".format(sum(rec_1s)/len(rec_1s) * 100))
    print (">> Unigram ROUGE-F: {:2.2f}".format(sum(f_1s)/len(f_1s) * 100))
    print (">> Bigram ROUGE-P: {:2.2f}".format(sum(prec_2s)/len(prec_2s) * 100))
    print (">> Bigram ROUGE-R: {:2.2f}".format(sum(rec_2s)/len(rec_2s) * 100))
    print (">> Bigram ROUGE-F: {:2.2f}".format(sum(f_2s)/len(f_2s) * 100))

def using_155_processed_data(input_dir):
    fpout_ref = open("tmp.ref", "w")
    fpout_hyp = open("tmp.hyp", "w")
    for i in range(0, 50):
        for line in open(input_dir + "/model/ref." + str(i) + ".txt"):
            line = line.strip()
            if line.startswith("<a name=\"1\">[1]</a> <a href=\"#1\" id=1>"):
                line = line.replace("<a name=\"1\">[1]</a> <a href=\"#1\" id=1>", "").replace("</a>", "")
                fpout_ref.write(line + "\n")
        for line in open(input_dir + "system/cand." + str(i) + ".txt"):
            line = line.strip()
            if line.startswith("<a name=\"1\">[1]</a> <a href=\"#1\" id=1>"):
                line = line.replace("<a name=\"1\">[1]</a> <a href=\"#1\" id=1>", "").replace("</a>", "")
                fpout_hyp.write(line + "\n")
    fpout_ref.close()
    fpout_hyp.close()

    # get rouge
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores("tmp.hyp", "tmp.ref", avg=True)
    for item in scores:
        print (item, scores[item])

if __name__ == '__main__':
    if sys.argv[1] == "get_docs":
        get_docs()
    elif sys.argv[1] == "pltrdy_rouge":
        ref_dir = "./50_docs/"
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_file_rouge(ref_dir, hyp_dir)
    elif sys.argv[1] == "pltrdy_hrouge":
        ref_dir = "./Bert_token_weight/"
        ref_tokens, ref_token_weights = get_token_weight(ref_dir)
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_file_hrouge(ref_tokens, ref_token_weights, hyp_dir)
    elif sys.argv[1] == "Hardy_rouge":
        ref_dir = "./50_docs/"
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Bert":
            hyp_dir = '../HROUGE_data/summaries/system_bert/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_hardy_rouge(ref_dir, hyp_dir)
    elif sys.argv[1] == "Hardy_hrouge_bert":
        ref_dir = "./Bert_token_weight/"
        ref_tokens, ref_token_weights = get_token_weight(ref_dir)
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_hardy_hrouge(ref_tokens, ref_token_weights, hyp_dir)
    elif sys.argv[1] == "Hardy_hrouge_glove":
        ref_dir = "./GloVE_token_weight/"
        ref_tokens, ref_token_weights = get_token_weight(ref_dir)
        if sys.argv[2] == "TConv":
            hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
        elif sys.argv[2] == "PT":
            hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
        elif  sys.argv[2] == "Ref":
            hyp_dir = '../HROUGE_data/summaries/ref_gold/'
        get_hardy_hrouge(ref_tokens, ref_token_weights, hyp_dir)
    elif sys.argv[1] == "test_from_155":
        using_155_processed_data("/tmp/tmp5m41ylwi/")
