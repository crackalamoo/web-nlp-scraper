import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
import string
import re

def get_tf_idf(page_list, compare_list=None):
    punct = string.punctuation + '“”‘’«»‹›‚„¡¿;،؛؟।॥•'
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
        print(df)
        print(df2)
        df = pd.concat([df, df2], axis=1).fillna(0)
        print(df)

    def get_tf(doc):
        tf = 0.5 + 0.5 * doc / np.max(doc)
        tf[doc == 0] = 0
        return tf
    tf = df.apply(get_tf, axis=0)
    idf = np.log((1+len(df.columns)) / (1+df.astype(bool).sum(axis=1)))
    df = tf.mul(idf, axis=0)

    return df

def topic_modeling(page_list, n_topics=5):
    df = get_tf_idf(page_list)

    lda = LatentDirichletAllocation(n_components=n_topics)
    term_doc = df.to_numpy().T
    lda.fit(term_doc)

    term_topics = lda.components_
    term_topics = pd.DataFrame(term_topics, columns=df.index)

    return term_topics

def get_top_words(page_list, compare_list=None):
    df = get_tf_idf(page_list, compare_list=compare_list)
    return df