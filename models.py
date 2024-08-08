import numpy as np
import pandas as pd
from collections import Counter
from utils import get_tf_idf, get_counts_df, split_sentences
from tqdm import tqdm

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
    import torch.nn as nn
    from torch.nn.functional import sigmoid
    from torch.utils.data import Dataset, DataLoader, random_split
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    device = torch.device(device)
    bert_model = bert_model.to(device)

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
            tokens = tokenizer(self.sentences[idx], return_tensors='pt').input_ids.to(device)
            label = torch.tensor(self.labels[idx], dtype=torch.int8, device=device)
            return tokens, label
    
    class ClassifierModel(nn.Module):
        def __init__(self):
            super(ClassifierModel, self).__init__()
            self.linear = nn.Linear(768, 1)
        
        def forward(self, x):
            x = self.linear(x)
            return x

    classifier = ClassifierModel()
    classifier = classifier.to(device)

    dataset = WebsiteDataset(page_list, compare_list)

    def collate_fn(batch):
        src_batch = []
        label_batch = []
        for src_sample, label_sample in batch:
            src_batch.append(src_sample[0, :])
            label_batch.append(label_sample)
        src = nn.utils.rnn.pad_sequence(src_batch)
        mask = (src != 0)
        labels = torch.tensor(label_batch, device=device)
        return src, mask, labels
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, collate_fn=collate_fn)

    for param in bert_model.parameters():
        param.requires_grad = False
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)

    n_epochs = 5
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}")

        train_losses = []
        for batch in tqdm(train_loader):
            inputs, mask, label = batch

            optimizer.zero_grad()

            bert_state = bert_model(inputs, attention_mask=mask).last_hidden_state
            bert_state = bert_state[-1, :, :] # get last token
            y_pred = classifier(bert_state)[:, 0] # shape [batch_size]
            y_true = label.to(dtype=y_pred.dtype)
            loss = loss_fn(y_pred, y_true)
            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())
        print("train loss:", np.mean(train_losses))

        with torch.no_grad():
            val_losses = []
            val_accs = []

            for batch in tqdm(val_loader):
                inputs, mask, label = batch
                bert_state = bert_model(inputs, attention_mask=mask).last_hidden_state
                bert_state = bert_state[-1, :, :] # get last token
                y_pred = classifier(bert_state)[:, 0] # shape [batch_size]
                y_true = label.to(dtype=y_pred.dtype)
                loss = loss_fn(y_pred, y_true)

                y_pred_r = torch.round(sigmoid(y_pred))
                acc = torch.sum(y_pred_r == y_true) / val_loader.batch_size

                val_losses.append(loss.item())
                val_accs.append(acc.item())
            print("val loss:", np.mean(val_losses))
            print("val acc:", np.mean(val_accs))