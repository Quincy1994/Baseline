# coding=utf-8
import gensim
from gensim.models import Doc2Vec
import numpy as np
from my_code.classifier import classify

class MYDM():

    def __init__(self):
        self.dm_model_path = ""
        self.dm_dim = 300
        self.dm = 1
        self.min_count = 5
        self.window = 5
        self.sample = 1e-3
        self.negative = 5
        self.workers = 4
        self.epochs = 20

    def train_dm(self, corpus):
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        train_doc = []
        for i, doc in enumerate(corpus):
            train_doc.append(TaggedDocument(doc, tags=[i]))
        model_dm = gensim.models.Doc2Vec(train_doc, min_count=self.min_count, size=self.dm_dim, sample=self.sample,negative=self.negative, workers=self.workers, dm=self.dm)
        print("=============== training DM model ===================")
        model_dm.train(train_doc, total_examples=model_dm.corpus_count)
        model_dm.save(self.dm_model_path)
        print("=================== DM model has been trained ==================")

    def get_dm_vectors(self, corpus):
        model = Doc2Vec.load(self.dm_model_path)
        texts = [document.split(" ") for document in corpus]
        dm_vectors = []
        for text in texts:
            v = model.infer_vector(text)
            dm_vectors.append(v)
        dm_vectors = np.array(dm_vectors)
        return dm_vectors

import pandas as pd
def train_dm_model():
    dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
    data = pd.read_csv(dataset, sep='\t')
    contents = data['weibo']
    contents = [str(content).lower().replace("|||", " ").split(" ") for content in contents]
    dm = MYDM()
    dm.dm_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/dm/dm.m"
    dm.train_dm(contents)

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
    label_path = "/media/iiip/Elements/数据集/user_profiling/weibo/label/age.csv"
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


# train_dm_model()
from sklearn.model_selection import StratifiedKFold

def train_test():
    contents, y = create_training_content()
    mydm = MYDM()
    mydm.dm_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/dm/dm.m"
    X_features = mydm.get_dm_vectors(contents)
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

