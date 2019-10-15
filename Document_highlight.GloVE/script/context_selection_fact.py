#coding=utf8

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

def facts_alignment(attn_dists, context, summary):
    tmp_attn = []
    for item in attn_dists:
        t_attn = []
        for i, term in enumerate(context):
            if label_classify(term) == "fact":
                t_attn.append(int(item[i]))
        tmp_attn.append(t_attn)
    default_len = len(tmp_attn[0])

    attn_dists = tmp_attn
    fact_attn_stack = []; label_stack = []; tokens = []
    for j, attn_f in enumerate(attn_dists):
        label_type = label_classify(summary[j])
        if label_type == "fact":
            tokens.append(summary[j])
            label_stack.append(label_type)
            fact_attn_stack.append(attn_dists[j])
        elif label_type == "end":
            pop_label = label_stack.pop()
            if pop_label == "fact":
                #fact_attn_stack.pop()
                tokens.append(summary[j])
        elif label_type == "token":
            tokens.append(summary[j])
        elif label_type == "reference":
            continue
        else:
            label_stack.append(label_type)
    return tokens, fact_attn_stack

def doc_fact(line):
    fact_stack = []
    type_stack = []
    fact_id_map = {}
    fact_terms = {}
    fact_id = 0
    for i, item in enumerate(line):
        l_type = label_classify(item)
        if l_type in ["fact"]:
            fact_stack.append(fact_id)
            fact_id_map[fact_id] = fact_stack[0]
            type_stack.append(l_type)
            fact_id += 1
        elif l_type in ["phrase"]:
            type_stack.append(l_type)
        elif l_type in ["end"]:
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_stack.pop()
        elif l_type not in ["reference"]:
            for j in fact_stack:
                if j not in fact_terms:
                    fact_terms[j] = []
                fact_terms[j].append(item)
    return fact_terms, fact_id_map

def doc_fact_mapping(line):
    fact_stack = []
    type_stack = []
    fact_id_map = {}
    fact_term_map = []
    terms = []
    fact_id = 0
    for i, item in enumerate(line):
        l_type = label_classify(item)
        if l_type in ["fact"]:
            fact_stack.append(fact_id)
            fact_id_map[fact_id] = fact_stack[0]
            type_stack.append(l_type)
            fact_id += 1
        elif l_type in ["phrase"]:
            type_stack.append(l_type)
        elif l_type in ["end"]:
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_stack.pop()
        elif l_type not in ["reference"]:
            fact_term_map.append([j for j in fact_stack])
            terms.append(item)
    return fact_term_map, terms, fact_id_map

def selected_facts(line, attn_dists):
    sentence = line.split(" ")
    rephrased = []
    for item in sentence:
        if label_classify(item) not in ["fact", "end"]:
            rephrased.append(item)
    slc_facts = set()
    for item in attn_dists:
        for i, label in enumerate(item):
            if label == 1:
                slc_facts.add(i)
    return " ".join(rephrased), slc_facts

def select_topn(attn_list, top_n):
    attn_dists = []
    for attn in attn_list:
        if top_n > 0:
            attn_np = np.array(attn)
            top_n_idx = attn_np.argsort()[-top_n:]
            attn_mask = np.zeros(len(attn))
            for max_id in top_n_idx:
                attn_mask[max_id] = 1
            attn_dists.append(list(attn_mask))
        else:
            attn_max = max(attn)
            if attn_max > 0:
                arr = np.array(attn) / attn_max
            else:
                arr = np.array(attn)
            if len(arr) >= 3:
                threshold_idx = arr.argsort()[-3]
            else:
                threshold_idx = arr.argsort()[-1]
            threshold = arr[threshold_idx]
            threshold = max(0.8, threshold)
            attn = [item if item >= threshold else 0 for item in arr]
            attn_dists.append(attn)
    return attn_dists

def one_file_attn_weight(filename, fpout_ctx, fpout_sum, top_n):
    for line in open(filename):
        json_obj = json.loads(line.strip())
        context = json_obj["context"]
        summary = json_obj["summary"]
        attn_dists = json_obj["attn_dists"]
        attn_dists = select_topn(attn_dists, top_n)
        rephrased, slc_facts = selected_facts(summary, attn_dists)
        doc_tokens = []
        for sentence in context.split("\t"):
            doc_tokens.extend(sentence.split(" "))
        fact_term_map, terms, fact_id_map = doc_fact_mapping(doc_tokens)
        merge_facts = set()
        for id in slc_facts:
            root = fact_id_map[id]
            if id == root:
                merge_facts.add(id)
            else:
                if root not in slc_facts:
                    merge_facts.add(id)
        tags = []
        for i, item in enumerate(terms):
            fact_ids = fact_term_map[i]
            tag = '0.00001'
            for id in fact_ids:
                if id in merge_facts:
                    tag = '1'
                    break
            tags.append(tag)
        fpout_ctx.write(" ".join(terms).encode("utf8") + "\n")
        fpout_sum.write(rephrased.encode("utf8") + "\t" + "|".join(tags) + "\n")

def one_file(filename, fpout_ctx, fpout_sum, top_n):
    for line in open(filename):
        json_obj = json.loads(line.strip())
        context = json_obj["context"]
        summary = json_obj["summary"]
        attn_dists = json_obj["attn_dists"]
        doc_tokens = []
        for sentence in context.split("\t"):
            doc_tokens.extend(sentence.split(" "))
        attn_dists = select_topn(attn_dists, top_n)
        rephrased, slc_facts = selected_facts(summary, attn_dists)
        fact_terms, fact_id_map = doc_fact(doc_tokens)
        merge_facts = set()
        for id in slc_facts:
            root = fact_id_map[id]
            if id == root:
                merge_facts.add(id)
            else:
                if root not in slc_facts:
                    merge_facts.add(id)
        context = [" ".join(fact_terms[id]) + " ." for id in merge_facts]
        fpout_ctx.write(" ".join(context).encode("utf8") + "\n")
        fpout_sum.write(rephrased.encode("utf8") + "\n")

if __name__ == '__main__':
    import sys
    import os
    import json
    import numpy as np
    if not os.path.exists("context_selection." + sys.argv[2]):
        os.system("mkdir context_selection." + sys.argv[2])
    fpout_ctx = open("context_selection." + sys.argv[2] + "/xsum_" + sys.argv[1] + "_src.jsonl", "w") 
    fpout_sum = open("context_selection." + sys.argv[2] + "/xsum_" + sys.argv[1] + "_tgt.jsonl", "w") 
    for fname in os.listdir("./tmp_output"):
        if fname.find(sys.argv[1]) > -1:
            #one_file("./tmp_output/" + fname, fpout_ctx, fpout_sum, int(sys.argv[2]))
            one_file_attn_weight("./tmp_output/" + fname, fpout_ctx, fpout_sum, int(sys.argv[2]))
    fpout_ctx.close()
    fpout_sum.close()
