import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
import string

def topic_modeling(data_dict, n_topics=5):

    matrix = {}
    for page in data_dict:
        title = page['title']
        words = page['body'].split()
        words = list(map(lambda s: s.translate(str.maketrans('', '', string.punctuation)).lower(), words))
        counts = Counter(words)
        matrix[title] = counts
    
    df = pd.DataFrame.from_dict(matrix).fillna(0)
    df[df == 1] = 0

    def get_tf(doc):
        tf = 0.5 + 0.5 * doc / np.max(doc)
        tf[doc == 0] = 0
        return tf
    tf = df.apply(get_tf, axis=0)
    idf = np.log((1+len(df.columns)) / (1+df.astype(bool).sum(axis=1)))
    df = tf.mul(idf, axis=0)

    lda = LatentDirichletAllocation(n_components=n_topics)
    term_doc = df.to_numpy().T
    lda.fit(term_doc)

    term_topics = lda.components_
    term_topics = pd.DataFrame(term_topics, columns=df.index)

    return term_topics