# coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from my_code.classifier import classify

class MYTFIDF():

    def __init__(self):
        self.max_feature = 50000         # max features of tfidf according to term-frequency
        self.min_df = 1
        self.ngram_range = (1, 1)
        self.tfidf_model_path = "./tfidf_model.m"  # take care of the store path


    def train_tfidf(self, corpus):

        """
        :param corpus: list
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("================== training tfidf model ==========================")
        tfidf = TfidfVectorizer(max_features=self.max_feature, min_df=self.min_df, ngram_range=self.ngram_range).fit(corpus)
        joblib.dump(tfidf, self.tfidf_model_path)
        print("=============== tfidf model has been trained =======================")

    def get_tfidf_vector(self, corpus):
        """
        :param corpus: list
            form of corpus:
                [   "I am good",
                    "How are you",
                    "That's OK"
                ]
        :return:
            tfidf vectors ---- shape(nrows, max_features)
        """
        tfidf_model = joblib.load(self.tfidf_model_path)
        tfidf_vectors = tfidf_model.transform(corpus).toarray()
        del tfidf_model
        return tfidf_vectors


# train tfidf model
def train_tfidf_model():
    dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
    data = pd.read_csv(dataset, sep='\t')
    contents = data['weibo']
    contents = [str(content).lower().replace("|||", " ") for content in contents]
    tfidf = MYTFIDF()
    tfidf.tfidf_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/tfidf_model.m"
    tfidf.train_tfidf(corpus=contents)

# create training content
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
    del user_content_dict
    return contents, y

# train and test (10-fold)
from sklearn.model_selection import StratifiedKFold
def train_test():
    contents, y = create_training_content()
    tfidf = MYTFIDF()
    tfidf.tfidf_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/tfidf_model.m"
    X_features = tfidf.get_tfidf_vector(corpus=contents)
    print(np.shape(X_features))
    del contents
    n_folds = 10
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    kf.get_n_splits(X_features, y)
    total_acc, total_pre, total_recall, total_macro_f1, total_micro_f1 = [], [], [], [], []
    for train_index, test_index in kf.split(X_features, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc, pre, recall, macro_f1, micro_f1 = classify(train_X=X_train,train_y=y_train, test_X=X_test, test_y=y_test)
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