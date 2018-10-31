# coding=utf-8
from gensim import corpora, models
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
import numpy as np
import pandas as pd
from my_code.classifier import classify


class MYLDA():
    def __init__(self):
        self.lda_dict_path = ""
        self.lda_model_path = ""
        self.lda_dim = 300

    def train_lda_model(self, corpus):
        """
        :param corpus:
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("====================== train lda model =======================")
        texts = [document.split(" ") for document in corpus]
        del corpus
        dictonary = corpora.Dictionary(texts)
        dictonary.save(self.lda_dict_path)
        bow_corpus = [dictonary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda = models.LdaModel(corpus_tfidf, id2word=dictonary, num_topics=self.lda_dim, iterations=1000)
        lda.save(self.lda_model_path)
        print("======================= lda  model has been trained ============================")

    def get_lda_vectors(self, corpus):
        """

        :param corpus:
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
             lda vectors ------ shape(nrows, lda_dim)
        """
        dictionary = corpora.Dictionary().load(self.lda_dict_path)
        lda = models.LdaModel(id2word=dictionary).load(self.lda_model_path)
        texts = [document.split(" ") for document in corpus]
        lda_vectors = np.zeros([len(texts), self.lda_dim], dtype=np.float32)
        for i, text in enumerate(texts):
            bow = dictionary.doc2bow(text)
            lda_value = lda[bow]
            for value in lda_value:
                d, v = value[0], value[1]
                lda_vectors[i][d] = v
        return lda_vectors

def train_lda_model():
    dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
    data = pd.read_csv(dataset, sep='\t')
    contents = data['weibo']
    contents = [str(content).lower().replace("|||", " ") for content in contents]
    del data
    lda = MYLDA()
    lda.lda_dict_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/lda/dict"
    lda.lda_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/lda/lda_model"
    lda.train_lda_model(contents)

# train_lda_model()

def create_training_content():
    dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
    data = pd.read_csv(dataset, sep='\t')
    contents = data['weibo']
    users = data['user']
    user_content_dict = {}
    for i in range(len(users)):
        user = users[i]
        content = contents[i]
        user_content_dict[user] = content
    label_path = "/media/iiip/Elements/数据集/user_profiling/weibo/label/gender.csv"
    data = pd.read_csv(label_path, sep='\t')
    labeled_user = data['user']
    labels = data['label']
    contents = []
    y = []
    for i in range(len(labeled_user)):
        user = labeled_user[i]
        contents.append(user_content_dict[user])
        y.append(int(labels[i]))
    y = np.array(y)
    contents = [str(content).lower().replace("|||", " ") for content in contents]
    del user_content_dict, users, data
    return contents, y

from sklearn.model_selection import StratifiedKFold

def train_test():
    contents, y = create_training_content()
    mylda = MYLDA()
    mylda.lda_dict_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/lda_50/dict"
    mylda.lda_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/lda_50/lda_model"
    X_features = mylda.get_lda_vectors(contents)
    del contents
    n_folds = 10
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    kf.get_n_splits(X_features, y)
    total_acc, total_pre, total_recall, total_macro_f1, total_micro_f1 = [], [], [], [], []
    for train_index, test_index in kf.split(X_features, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc, pre, recall, macro_f1, micro_f1 = classify(train_X=X_train, train_y=y_train, test_X=X_test, test_y=y_test)
        total_acc.append(acc)
        total_pre.append(pre)
        total_recall.append(recall)
        total_macro_f1.append(macro_f1)
        total_micro_f1.append(micro_f1)
        del X_train, X_test, y_train, y_test
    print("======================")
    print("avg acc:", np.mean(total_acc))
    print("avg pre:", np.mean(total_pre))
    print("avg recall:", np.mean(total_recall))
    print("avg macro_f1:", np.mean(total_macro_f1))
    print("avg micro_f1:", np.mean(total_micro_f1))
    print("======================")

train_test()