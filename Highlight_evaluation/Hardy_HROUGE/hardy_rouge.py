from nltk.util import ngrams
from itertools import chain
from collections import Counter
import json

def _beta(n, text, weights):
    weight_dict = {}; count_dict = {}
    for i, item in enumerate(text):
        pair = item
        if pair not in weight_dict:
            weight_dict[pair] = sum(weights[i]) / n
            count_dict[pair] = 1
        else:
            weight_dict[pair] += sum(weights[i]) / n
            count_dict[pair] += 1
    for pair in weight_dict:
        weight_dict[pair] = weight_dict[pair] / count_dict[pair]
    return weight_dict

def HR_rec(n, S, D, H=None):
    n_gram_D = D.split()
    n_gram_D = list(ngrams(n_gram_D, n))
    count_n_gram_D = Counter(n_gram_D)

    n_gram_S = list(ngrams(S.split(), n))
    count_n_gram_S = Counter(n_gram_S)
    n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))

    n_gram_H = list(ngrams(H, n))
    beta = _beta(n, n_gram_D, n_gram_H)

    numerator = 0
    for g in n_gram_DnS:
        numerator += beta[g] * min(count_n_gram_D[g], count_n_gram_S[g])
    denominator = 0
    for g in set(n_gram_D):
        denominator += beta[g] * count_n_gram_D[g]
    return numerator/max(denominator, 1)


def HR_prec(n, S, D, H=None):
    n_gram_D = D.split()
    n_gram_D = list(ngrams(n_gram_D, n))
    count_n_gram_D = Counter(n_gram_D)

    n_gram_S = list(ngrams(S.split(), n))
    count_n_gram_S = Counter(n_gram_S)
    n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))

    n_gram_H = list(ngrams(H, n))
    beta = _beta(n, n_gram_D, n_gram_H)

    numerator = 0
    for g in n_gram_DnS:
        numerator += beta[g] * min(count_n_gram_D[g], count_n_gram_S[g])
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
