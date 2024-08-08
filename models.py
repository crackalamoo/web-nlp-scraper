import numpy as np
import pandas as pd
from collections import Counter
from utils import get_tf_idf, get_counts_df, split_sentences

def topic_modeling(page_list, compare_list=None, n_topics=5):
    from sklearn.decomposition import LatentDirichletAllocation
    tfidf = get_tf_idf(page_list)
    if compare_list is not None:
        df2 = get_tf_idf(compare_list)
        tfidf = pd.concat([tfidf, df2], axis=1).fillna(0)
        tfidf.columns = ['Loaded', 'Comparison']

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
        tfidf.columns = ['Loaded', 'Comparison']
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

def stopwords_tf_idf(page_list, compare_list=None, remove_stop=True, no_num_stop=True, pipeline='en_core_web_sm'):
    import spacy
    counts_df = get_counts_df(page_list)
    if compare_list is not None:
        counts_df = counts_df.sum(axis=1)
        df2 = get_counts_df(compare_list).sum(axis=1)
        counts_df = pd.concat([counts_df, df2], axis=1).fillna(0)
        counts_df.columns = ['Loaded', 'Comparison']

    nlp = spacy.load(pipeline)
    to_delete = []
    for word in counts_df.index:
        word_doc = nlp(word)
        is_stop = False
        if len(word_doc) == 1:
            is_stop = word_doc[0].is_stop
        if no_num_stop:
            # do not consider numerals to be stopwords
            is_stop = is_stop and not word_doc[0].like_num
            is_stop = is_stop and not any(c.isdigit() for c in word)
        
        if remove_stop and is_stop:
            to_delete.append(word)
        elif not remove_stop and not is_stop:
            to_delete.append(word)
    
    counts_df = counts_df.drop(to_delete, axis=0)
    tf_idf = get_tf_idf(page_list, counts_df=counts_df)
    return tf_idf

def bert_classifier(page_list, compare_list):
    from transformers import BertTokenizer, BertModel
    import torch
    from torch.utils.data import Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    class WebsiteDataset(Dataset):
        def __init__(self, pl, cl):
            page_snt = split_sentences(pl)
            compare_snt = split_sentences(cl)
            min_len = min(len(page_snt), len(compare_snt))
            page_snt = page_snt[:min_len]
            compare_snt = compare_snt[:min_len]
            self.labels = [0] * len(page_snt)
            self.labels += [1] * len(compare_snt)
            self.sentences = page_snt + compare_snt
        
        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            tokens = tokenizer(self.sentences[idx], return_tensors='pt')
            label = torch.tensor(self.labels[idx], dtype=torch.int8)
            return tokens, label
    
    dataset = WebsiteDataset(page_list, compare_list)