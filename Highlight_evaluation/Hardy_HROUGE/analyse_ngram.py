from nltk.util import ngrams
from itertools import chain
from collections import Counter
import json

MAX_LEN = 4

delete_sentences = ["These are external links and will open in a new window",
        "Share this with",
        "Email",
        "Facebook",
        "Messenger",
        "Twitter",
        "Pinterest",
        "WhatsApp",
        "LinkedIn",
        "Linkedin",
        "Copy this link"]

def _parse(toks):
    components = []
    for idx, token in enumerate(toks):
        aWord = token.lower()
        if aWord == '-lrb-':
            aWord = '('
        elif aWord == '-rrb-':
            aWord = ')'
        elif aWord == '``':
            aWord = '"'
        elif aWord == '\'\'':
            aWord = '"'
        elif aWord == '#':
            aWord = '£'
        #elif aWord == '€':
        #    aWord = '$'
        #if aWord.endswith("km") and aWord != "km":
        #    components.append(aWord.replace("km", ""))
        #    components.append("km")
        #elif aWord.endswith("cm") and aWord != "cm":
        #    components.append(aWord.replace("cm", ""))
        #    components.append("cm")
        #else:
        components.append(aWord)
    return components

def rephrase(line):
    flist = [item for item in line.strip().split("\t") if (item not in delete_sentences)]
    new_line = " ".join(flist)
    return " ".join(_parse(new_line.split()))

def numH(w, H):
    result = 0
    for h in H:
        h_words = list(chain(*[x.split() for x in h]))
        if w in h_words:
            result += len(h_words) / MAX_LEN
    return result


def beta(n, g, w, H):
    numerator = 0
    denominator = 0
    m = len(w)
    for i in range(m-n+1):
        total_NumH = 0
        for j in range(i, i+n):
            if w[i:i+n] == list(g):
                total_NumH += numH(w[j], H)
        total_NumH /= len(H)
        total_NumH /= n
        numerator += total_NumH
    for i in range(m-n+1):
        if w[i:i+n] == list(g):
            denominator += 1
    return (numerator + 1)/(denominator + 1) - 1


def R_rec(n, S, D, H=None):
    n_gram_D = rephrase(D).split()
    #n_gram_D = D.split()
    n_gram_D = list(ngrams(n_gram_D, n))
    count_n_gram_D = Counter(n_gram_D)
    #n_gram_S = list(ngrams(rephrase(S).split(), n))
    n_gram_S = list(ngrams(S.split(), n))
    count_n_gram_S = Counter(n_gram_S)

    n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))
    numerator = 0
    for g in n_gram_DnS:
        if H is not None:
            numerator += beta(n, g, D, H) * min(count_n_gram_D[g], count_n_gram_S[g])
        else:
            numerator += 1 * min(count_n_gram_D[g], count_n_gram_S[g])
    denominator = 0
    for g in set(n_gram_D):
        if H is not None:
            denominator += beta(n, g, D, H) * count_n_gram_D[g]
        else:
            denominator += 1 * count_n_gram_D[g]
    return numerator/max(denominator, 1)


def R_prec(n, S, D, H=None):
    n_gram_D = rephrase(D).split()
    #n_gram_D = D.split()
    n_gram_D = list(ngrams(n_gram_D, n))
    count_n_gram_D = Counter(n_gram_D)
    #n_gram_S = list(ngrams(rephrase(S).split(), n))
    n_gram_S = list(ngrams(S.split(), n))
    count_n_gram_S = Counter(n_gram_S)
    n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))
    numerator = 0
    for g in n_gram_DnS:
        if H is not None:
            numerator += beta(n, g, D, H) * min(count_n_gram_D[g], count_n_gram_S[g])
        else:
            numerator += 1 * min(count_n_gram_D[g], count_n_gram_S[g])
    denominator = 0
    for g in set(n_gram_S):
        denominator += 1 * count_n_gram_S[g]
    return numerator/max(denominator, 1)


if __name__ == '__main__':
    recs = []; precs = []
    for line in open("highres/debug.data"):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        doc = json_obj["document"]
        summary = json_obj["summary"]
        rec = R_rec(1, summary, doc)
        recs.append(rec)
        prec = R_prec(1, summary, doc)
        precs.append(prec)
        print(doc_id, rec, prec)
    print (sum(recs)/len(recs))
    print (sum(precs)/len(precs))
