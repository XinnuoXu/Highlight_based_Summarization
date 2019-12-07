#coding=utf8

import os, sys

def rewrite(toks):
        qoute_num = 0
        breaket_num = 0
        for i, tok in enumerate(toks):
                if tok == "\"":
                        qoute_num += 1
                        if qoute_num%2:
                                toks[i] = "``"
                        else:
                                toks[i] = "\'\'"
                if tok == "@@":
                        breaket_num += 1
                        if breaket_num%2:
                                toks[i] = "-lrb-"
                        else:
                                toks[i] = "-rrb-"
        return toks

if __name__ == '__main__':
        ref = {}
        for filename in os.listdir("50_trees"):
                if filename.endswith("src"):
                        continue
                doc_id = filename.split(".")[0]
                with open("50_trees/" + filename, 'r') as file:
                        ref[doc_id] = file.read().strip()

        for doc_id in ref:
                items = ref[doc_id].split()
                new_toks = []
                for tok in items:
                        if not (tok[0] == '(' or tok[0] == ')' or tok.startswith("*trace-") or tok == "£"):
                                new_toks.append(tok)
                ref[doc_id] = " ".join(rewrite(new_toks))

        file_gold = [line.strip() for line in open(sys.argv[1])]
        file_cand = [line.strip() for line in open(sys.argv[2])]
        
        mapping = {}
        for i, gold in enumerate(file_gold):
                mapping[gold] = file_cand[i]

        if not os.path.exists(sys.argv[3]):
                os.system("mkdir " + sys.argv[3])

        for doc_id in ref:
                if ref[doc_id] in mapping:
                        fpout = open(sys.argv[3] + "/" + doc_id + ".data", "w")
                        fpout.write(mapping[ref[doc_id]] + "\n")
                        fpout.close()
