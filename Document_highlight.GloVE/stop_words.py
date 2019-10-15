#coding=utf8

if __name__ == '__main__':
    doc_freq = {}
    for line in open("corpus.txt"):
        flist = set(line.strip().split(" "))
        for item in flist:
            if item not in doc_freq:
                doc_freq[item] = 1
            else:
                doc_freq[item] += 1
    for item in sorted(doc_freq.items(), key = lambda d:d[1], reverse = True):
        print item[1], item[0]

