#coding=utf8

import os
import sys
import json

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

def eva_highlight_v1(line):
    tokens = [[] for i in enumerate(line)]; names = []
    fact_name_stack = []
    id_stack = []
    type_stack = []
    only_tokens = []
    for i, item in enumerate(line):
        l_type = label_classify(item)
        if l_type in ["fact", "phrase"]:
            tokens[i].append("<strong>")
            id_stack.append(i)
            type_stack.append(l_type)
            if l_type == "fact":
                fact_name_stack.append(item[1:])
                names.append(item[1:])
            else:
                names.append(fact_name_stack[-1] + "|||" + item[1:])
        elif l_type == "end":
            pop_id = id_stack.pop()
            tokens[pop_id].append("</strong>")
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_name_stack.pop()
            names.append("")
        else:
            if l_type == "reference":
                names.append("")
            else:
                for j in range(len(tokens)):
                    tokens[j].append(item)
                names.append("")
                only_tokens.append(item)

    ret_list = []; ret_name = []
    for i, item in enumerate(tokens):
        if names[i] != "":
            ret_list.append(item)
            ret_name.append(names[i])
    return ret_list, ret_name, only_tokens

def eva_highlight(line):
    tokens = [[] for i in enumerate(line)]; names = []
    length = len(line)
    fact_name_stack = []
    fact_id_stack = []
    phrase_id_stack = []
    type_stack = []
    only_tokens = []
    for i, item in enumerate(line):
        l_type = label_classify(item)
        if l_type in ["fact", "phrase"]:
            type_stack.append(l_type)
            if l_type == "fact":
                fact_name_stack.append(item[1:])
                fact_id_stack.append(i)
                names.append(item[1:])
            else:
                phrase_id_stack.append(i)
                names.append(fact_name_stack[-1] + "|||" + item[1:])
        elif l_type == "end":
            pop_type = type_stack.pop()
            if pop_type == "fact":
                fact_name_stack.pop()
                fact_id_stack.pop()
            if pop_type == "phrase":
                phrase_id_stack.pop()
            names.append("")
        else:
            if l_type == "reference":
                names.append("")
            else:
                for j in range(length):
                    if len(phrase_id_stack) > 0 and \
                            j == phrase_id_stack[-1] and \
                            len(type_stack) > 0 and \
                            type_stack[-1] == "phrase":
                        tokens[j] = tokens[j] + ["<strong>", item, "</strong>"]
                        continue
                    if len(fact_id_stack) > 0 and j == fact_id_stack[-1]:
                        tokens[j] = tokens[j] + ["<strong>", item, "</strong>"]
                        continue
                    tokens[j] = tokens[j] + [item]
                names.append("")
                only_tokens.append(item)

    ret_list = []; ret_name = []
    for i, item in enumerate(tokens):
        if names[i] != "":
            ret_list.append(item)
            ret_name.append(names[i])
    return ret_list, ret_name, only_tokens

def highlight_filter(toks, stop_words):
    b_strong = False
    non_stop_toks = []
    for tok in toks:
        if tok == "<strong>":
            b_strong = True
            continue
        if tok == "</strong>":
            b_strong = False
            continue
        if b_strong:
            if (tok not in stop_words) and len(tok)>1:
                non_stop_toks.append(tok)
    if len(non_stop_toks) == 0:
        return True
    return False

if __name__ == '__main__':
    input_dir = "Bert_highlight/"
    output_dir = "human_eva/"

    stop_words = [term.strip() for term in open("./stop_words.txt")]

    for filename in os.listdir(input_dir):
        fid = filename.split(".")[0]
        with open(input_dir + filename, 'r') as file:
            line = file.read().strip()
        json_obj = json.loads(line)
        summ = json_obj['decoded_lst']
        ret_list, ret_name, only_tokens = eva_highlight(summ)

        fpout = open(output_dir + fid + ".data", "w")
        res_json = []
        res_json.append({"name": "Full", "text": "<strong> " + " ".join(only_tokens) + " </strong>"})
        '''
        for i, item in enumerate(ret_list):
            if highlight_filter(ret_list[i], stop_words):
                continue
            res_json.append({"name": ret_name[i], "text": " ".join(ret_list[i])})
        '''
        fpout.write(json.dumps(res_json))
        fpout.close()
