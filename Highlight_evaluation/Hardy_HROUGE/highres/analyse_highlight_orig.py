#%%
"""
Extract results from database and then perform analysis on top of to draw insights
"""
import json
import os
from nltk.util import ngrams
from nltk.stem.porter import *
from collections import Counter
from itertools import chain
import pandas as pd
from flask_sqlalchemy import SQLAlchemy

from backend.models import Dataset, Document, Summary, SummaryGroup, AnnotationResult, DocStatus
from backend.app import create_app

stemmer = PorterStemmer()

#%%
# Loading data from database
summary_name = 'BBC_system_ptgen'
#summary_name = 'BBC_system_tconvs2s'
#summary_name = 'BBC_ref_gold'
app = create_app()
db = SQLAlchemy(app)
results_dir = './results/'
q_results = db.session.query(Summary, SummaryGroup, Document, Dataset) \
    .join(Document).join(SummaryGroup).join(Dataset) \
    .filter(Dataset.name == 'BBC', SummaryGroup.name == summary_name).all()


#%%
# Process data from database into components and components' type
def parse(doc_json):
    """
    Parse document into components (list of all tokens) and comp_types (list of types for all tokens)
    """
    components = []
    comp_types = []
    index = []
    for sent in doc_json['sentences']:
        for idx, token in enumerate(sent['tokens']):
            aWord = token['word'].lower()
            if token['word'] == '-LRB-':
                aWord = '('
            elif token['word'] == '-RRB-':
                aWord = ')'
            elif token['word'] == '``':
                aWord = '"'
            elif token['word'] == '\'\'':
                aWord = '"'
            components.append(aWord)
            if aWord.strip() == '':
                comp_types.append('whitespace')
            else:
                comp_types.append('word')
            index.append(len(index))
            if idx != len(sent['tokens']) - 2:
                components.append(' ')
                comp_types.append('whitespace')
                index.append(len(index))
    data = {
        'content': pd.Series(components),
        'type': pd.Series(comp_types),
        'index': pd.Series(index),
    }
    return pd.DataFrame(data)

# Contains information of each word in the document
df_doc_prop = pd.DataFrame([])

summaries = {}
for summ, _, doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    df_doc_prop = df_doc_prop.append(parse(doc_json).assign(doc_id=doc_json['doc_id']))
    summaries[doc.doc_id] = summ.text.split()

# Contains data of the document with the summary
df_doc = pd.DataFrame(df_doc_prop[df_doc_prop['type'] != 'whitespace'].groupby('doc_id').count())
df_doc = df_doc.rename(columns={'index': 'word_len'}).drop(columns=['type', 'content'])
df_doc['summ'] = df_doc.index.map(summaries.get)

#%%
# Retrieve highlights
def process_doc(doc_json, word_idx):
    """
    Build indexes and texts for the given document
    """
    indexes = []
    texts = []
    result_ids = []
    results = doc_json['results']
    doc_id = doc_json['doc_id']
    for result_id, data in results.items():
        for h_id, highlight in data['highlights'].items():
            if highlight['text'] == '':
                continue
            word_only_highlight = [idx for idx in highlight['indexes'] if word_idx.loc[idx]['type'] == 'word']
            indexes.append(word_only_highlight)
            texts.append(highlight['text'].lower())
            result_ids.append(result_id)
    data = {
        'indexes': pd.Series(indexes),
        'text': pd.Series(texts),
        'result_id': pd.Series(result_ids),
        'doc_id': doc_id
    }
    return pd.DataFrame(data)

df_h = pd.DataFrame([])
for summ, _, doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    word_idx = df_doc_prop.groupby('doc_id').get_group(doc_json['doc_id'])
    df_h = df_h.append(process_doc(doc_json, word_idx))

#%%
# Retrieve result per coder and store them in DataFrame
q_results = db.session.query(Document).all()

df_annotations = pd.DataFrame([])
df_doc_prop_group = df_doc_prop.groupby('doc_id')
count = 0
test = {}
for doc in q_results:
    results = json.loads(doc.doc_json)['results']
    for result_id, result in results.items():
        highlights = result['highlights']
        indexes = []
        texts = []
        word_idx = df_doc_prop_group.get_group(doc.doc_id)
        for key, highlight in highlights.items():
            if highlight['text'] == '':
                continue
            word_only_highlight = [idx for idx in highlight['indexes'] if word_idx.loc[idx]['type'] == 'word']
            indexes.append(word_only_highlight)
            texts.append(highlight['text'])
        df_annotations = df_annotations.append(
            pd.DataFrame({
                'indexes': pd.Series(indexes),
                'texts': pd.Series(texts)
            }).assign(doc_id=doc.doc_id, result_id=result_id))
#%%
# Modified n-gram
df_doc = df_doc.assign(doc_text=lambda x: df_doc_prop[df_doc_prop['type'] == 'word'].groupby('doc_id')['content'].apply(list))
df_doc = df_doc.assign(doc_text_join=lambda x: df_doc['doc_text'].apply(' '.join))
# df_doc = df_doc.assign(h_text_join=lambda x: df_doc['h_text'].apply(' '.join))
df_doc = df_doc.assign(h_idxs=lambda x: df_h.groupby('doc_id').apply(lambda x: list(set(chain(*x.indexes)))))

df_doc = df_doc.assign(doc_idxs=lambda x: df_doc_prop[df_doc_prop['type']=='word'].groupby('doc_id')['index'].apply(list))

df_doc = df_doc.assign(h_len=lambda x: df_doc[['h_idxs']].apply(lambda x: len(x['h_idxs']), axis=1))

MAX_LEN = 30
df_ngrams = pd.DataFrame([])

#%%
def numH(w, H):
    result = 0
    H_group = H.groupby('result_id')
    highlights = {}
    for result_id, data in H_group:
        if result_id not in highlights.keys():
            highlights[result_id] = data['indexes']
        else:
            highlights[result_id].append(data['indexes'])
    for result_id in highlights.keys():
        h_words = list(chain(*[highlight for highlight in highlights[result_id]]))
        if w in h_words:
            result += len(h_words) / MAX_LEN
    return result


def beta(n, g, w, H):
    numerator = 0
    denominator = 0
    m = len(w[0])
    for i in range(m-n+1):
        total_NumH = 0
        for j in range(i, i+n):
            if w[0][i:i+n] == list(g):
                total_NumH += numH(w[1][j], H)
        total_NumH /= 10
        total_NumH /= n
        numerator += total_NumH
    for i in range(m-n+1):
        if w[0][i:i+n] == list(g):
            denominator += 1
    if denominator == 0 or numerator == 0:
        return 0
    return numerator/denominator


def R_rec(n, S, D, H):
    n_gram_D = list(ngrams(D[0], n))
    count_n_gram_D = Counter(n_gram_D)
    n_gram_S = list(ngrams(S, n))
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


def R_prec(n, S, D, H):
    n_gram_D = list(ngrams(D[0], n))
    count_n_gram_D = Counter(n_gram_D)
    n_gram_S = list(ngrams(S, n))
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
        # denominator += beta(n, g, D, H) * count_n_gram_S[g]
        denominator += 1 * count_n_gram_S[g]
    return numerator/max(denominator, 1)


df_h_g = df_h.groupby('doc_id')
recs_1 = []
recs_2 = []
precs_1 = []
precs_2 = []
f_1s_1 = []
f_1s_2 = []
doc_ids = []
for doc_id, data in df_annotations.groupby('doc_id'):
    print(doc_id)
    summ = db.session.query(Summary, SummaryGroup, Document) \
        .join(Document).join(SummaryGroup) \
        .filter(
        Dataset.name == 'BBC',
        SummaryGroup.name == summary_name,
        Document.doc_id == doc_id) \
        .first()[0]
    doc_texts = (list(df_doc.loc[doc_id]['doc_text']), list(df_doc.loc[doc_id]['doc_idxs']))
    debug_obj = {}
    debug_obj['doc_id'] = doc_id
    debug_obj['document'] = " ".join(doc_texts[0])
    debug_obj['summary'] = summ.text
    #print (json.dumps(debug_obj))
    H = df_h_g.get_group(doc_id)
    import math
    print('Calculating 1-gram')
    r_1 = R_rec(1, summ.text.split(), doc_texts, H)
    p_1 = R_prec(1, summ.text.split(), doc_texts, H)
    if r_1 + p_1 == 0:
        f_1_1 = 0
    else:
        f_1_1 = 2 * r_1 * p_1 / (r_1 + p_1)
    print('Calculating 2-gram')
    r_2 = R_rec(2, summ.text.split(), doc_texts, H)
    p_2 = R_prec(2, summ.text.split(), doc_texts, H)
    if r_2 + p_2 == 0:
        f_1_2 = 0
    else:
        f_1_2 = 2 * r_2 * p_2 / (r_2 + p_2)
    recs_1.append(r_1)
    recs_2.append(r_2)
    precs_1.append(p_1)
    precs_2.append(p_2)
    f_1s_1.append(f_1_1)
    f_1s_2.append(f_1_2)
    doc_ids.append(doc_id)
df_f_1 = pd.DataFrame({
    'doc_id': pd.Series(doc_ids),
    'recalls_1': pd.Series(recs_1),
    'precisions_1': pd.Series(precs_1),
    'f_1s_1': pd.Series(f_1s_1),
    'recalls_2': pd.Series(recs_2),
    'precisions_2': pd.Series(precs_2),
    'f_1s_2': pd.Series(f_1s_2)
})

#%%
# Save to file
df_f_1.to_csv(os.path.join(results_dir, '%s_rouge.csv' % summary_name))
df_f_1.describe().to_csv(os.path.join(results_dir, '%s_rouge_describe.csv' % summary_name))

