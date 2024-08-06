import numpy as np
import pandas as pd
from collections import Counter
import string
import re

def get_tf_idf(page_list, compare_list=None):
    punct = string.punctuation + '“”‘’«»‹›‚„¡¿;،؛۔؟।॥•·።'
    def get_counts_df(pl):
        matrix = {}
        for page in pl:
            title = page['title'] if 'title' in page else page['url']
            body = re.sub(r'[-‐‑‒–—―]', ' ', page['body'])
            words = body.split()
            words = list(map(lambda s: s.translate(str.maketrans('', '', punct)).lower(), words))
            counts = Counter(words)
            matrix[title] = counts
        df = pd.DataFrame.from_dict(matrix).fillna(0)
        return df

    df = get_counts_df(page_list)
    if compare_list is not None:
        df = df.sum(axis=1)
        df2 = get_counts_df(compare_list).sum(axis=1)
        df = pd.concat([df, df2], axis=1).fillna(0)

    def get_tf(doc):
        tf = 0.5 + 0.5 * doc / np.max(doc)
        tf[doc == 0] = 0
        return tf
    tf = df.apply(get_tf, axis=0)
    idf = np.log((1+len(df.columns)) / (1+df.astype(bool).sum(axis=1)))
    df = tf.mul(idf, axis=0)

    return df

def topic_modeling(page_list, compare_list=None, n_topics=5):
    from sklearn.decomposition import LatentDirichletAllocation
    tfidf = get_tf_idf(page_list)
    if compare_list is not None:
        df2 = get_tf_idf(compare_list)
        tfidf = pd.concat([tfidf, df2], axis=1).fillna(0)

    lda = LatentDirichletAllocation(n_components=n_topics)
    term_doc = tfidf.to_numpy().T
    lda.fit(term_doc)

    term_topics = lda.components_
    term_topics = pd.DataFrame(term_topics, columns=tfidf.index)

    return term_topics

def get_similarities(page_list, compare_list=None):
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf = get_tf_idf(page_list)
    if compare_list is not None:
        df2 = get_tf_idf(compare_list)
        compare_cutoff = len(tfidf.columns)
        tfidf = pd.concat([tfidf, df2], axis=1).fillna(0)
    sims = cosine_similarity(tfidf.to_numpy().T)
    sims = pd.DataFrame(sims, index=tfidf.columns, columns=tfidf.columns)

    tri = np.tri(sims.values.shape[0], sims.values.shape[0], k=-1)
    sims.values[tri == 0] = np.nan
    pairs = []
    diff_pairs = []
    for i1, c1 in enumerate(sims.index):
        for i2, c2 in enumerate(sims.columns):
            if compare_list is not None and ((i1 < compare_cutoff and i2 >= compare_cutoff) or (i2 < compare_cutoff and i1 >= compare_cutoff)):
                diff_pairs.append(len(pairs))
            pairs.append(f'{c1} <-> {c2}')
    sims_list = pd.Series(sims.values.flatten(), index=pairs)
    if compare_list is not None:
        diff_pairs = sims_list.iloc[diff_pairs].dropna()
    sims_list = sims_list.dropna()

    if compare_list is not None:
        return sims, sims_list, diff_pairs
    return sims, sims_list


def get_top_words(page_list, compare_list=None):
    tfidf = get_tf_idf(page_list, compare_list=compare_list)
    return tfidf


def named_entity_recognition(page_list, pipeline='en_core_web_sm'):
    import spacy
    nlp = spacy.load(pipeline)
    counts = Counter()
    exclude = set(['CARDINAL', 'ORDINAL', 'PERCENT', 'MONEY', 'QUANTITY',
                   'DATE', 'TIME'])
    for page in page_list:
        entities = []
        for word in nlp(page['body']).ents:
            if word.label_ not in exclude:
                entities.append((word.text, word.label_))
        counts += Counter(entities)
    return counts