#coding=utf8

import sys, os
import json

def extract_loss(obj):
    return obj['ex_loss']

if __name__ == '__main__':
    attn_path = "../Summarization/logs/attn.log"
    with open(attn_path, 'r') as file:
        json_objs = json.loads(file.read().strip())
    sorted_json = sorted(json_objs, key = lambda i: i['ex_loss'])
    best_ex = sorted_json[0]
    worst_ex = sorted_json[-1]
    best_fpout = open("./best_attn.json", "w")
    best_fpout.write(json.dumps(best_ex))
    worst_fpout = open("./worst_attn.json", "w")
    worst_fpout.write(json.dumps(worst_ex))
    best_fpout.close()
    worst_fpout.close()
