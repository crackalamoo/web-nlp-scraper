import pandas as pd
import numpy as np
import string
import re
from collections import Counter

def get_counts_df(pl):
    punct = string.punctuation + '“”‘’«»‹›‚„¡¿;،؛۔؟।•·።。！？'
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

def get_tf_idf(page_list, compare_list=None, counts_df=None):
    if counts_df is None:
        df = get_counts_df(page_list)
        if compare_list is not None:
            df = df.sum(axis=1)
            df2 = get_counts_df(compare_list).sum(axis=1)
            df = pd.concat([df, df2], axis=1).fillna(0)
    else:
        df = counts_df

    def get_tf(doc):
        tf = 0.5 + 0.5 * doc / np.max(doc)
        tf[doc == 0] = 0
        return tf
    tf = df.apply(get_tf, axis=0)
    idf = np.log((1+len(df.columns)) / (1+df.astype(bool).sum(axis=1)))
    df = tf.mul(idf, axis=0)

    return df

def split_sentences(page_list, max_length=250):
    sentences = []
    stops = '.!?।۔؟።。！？'
    for page in page_list:
        sentences += re.findall(f'.+?[{stops}]\s+', page['body'])
    for i in reversed(range(len(sentences))):
        prev = sentences[i]
        if len(sentences[i]) > max_length:
            while len(sentences[i]) > max_length and sentences[i].rfind(' ') != -1:
                sentences[i] = sentences[i][:sentences[i].rfind(' ')]
            sentences[i] = sentences[i][:max_length]
        sentences[i] = sentences[i].strip()
        if len(sentences[i]) == 0:
            sentences.pop(i)
    return sentences