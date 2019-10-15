#coding=utf8

if __name__ == '__main__':
    import sys
    docs = []
    for line in open("context_selection.1/xsum_" + sys.argv[1] + "_src.jsonl"):
        docs.append(line.strip().split(" "))

    targets = []; sources = []
    for i, line in enumerate(open("context_selection.1/xsum_" + sys.argv[1] + "_tgt.jsonl")):
        line = line.strip()
        flist = line.split("\t")
        targets.append(flist[0])
        tags = flist[1].split("|"); doc = docs[i]
        selected = []
        for j, term in enumerate(doc):
            if tags[j] == "1":
                selected.append(term)
        sources.append(" ".join(selected))

    fpout_t = open("targets.txt", "w")
    for t in targets:
        fpout_t.write(t + "\n")
    fpout_t.close()

    fpout_s = open("sources.txt", "w")
    for s in sources:
        fpout_s.write(s + "\n")
    fpout_s.close()
