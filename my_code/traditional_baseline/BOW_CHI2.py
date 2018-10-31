# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
import pandas as pd
from my_code.classifier import classify

class MYCHI2():

    def __init__(self):
        self.chi2_max_feature = 50000         # max features of tfidf according to term-frequency
        self.cv_model_path = ""
        self.chi2_model_path = ""
        self.cv_max_feature = 100000


    def train_chi2(self, corpus, label_y):

        """
        :param corpus: list
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("================== train CountVector model =======================")
        cv = CountVectorizer(max_features=self.cv_max_feature).fit(corpus)
        print("==================== CountVector model has been trained =================")
        X_features = cv.transform(corpus).toarray()
        chi2_model = SelectKBest(chi2, k=self.chi2_max_feature).fit(X_features, label_y)
        print("===================== chi2 model has been trained =====================")
        joblib.dump(cv, self.cv_model_path)
        joblib.dump(chi2_model, self.chi2_model_path)
        print("=============== models have been trained =======================")
        del X_features, chi2_model, cv

    def get_chi2_vector(self, corpus):
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
        cv_model = joblib.load(self.cv_model_path)
        chi2_model = joblib.load(self.chi2_model_path)
        cv_vectors = cv_model.transform(corpus)
        chi2_vectors = chi2_model.transform(cv_vectors)
        del cv_model, chi2_model
        return chi2_vectors

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
def train_chi2_model():
    contents, y = create_training_content()
    contents = np.array(contents)
    chi2 = MYCHI2()
    chi2.cv_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/cv_model.m"
    chi2.chi2_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/chi2_model.m"
    n_folds = 10
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    kf.get_n_splits(contents, y)
    total_acc, total_pre, total_recall, total_macro_f1, total_micro_f1 = [], [], [], [], []
    for train_index, test_index in kf.split(contents, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = contents[train_index], contents[test_index]
        y_train, y_test = y[train_index], y[test_index]
        chi2.train_chi2(X_train, y_train)
        X_chi2_train = chi2.get_chi2_vector(X_train)
        X_chi2_test = chi2.get_chi2_vector(X_test)
        del X_train, X_test
        acc, pre, recall, macro_f1, micro_f1 = classify(train_X=X_chi2_train, train_y=y_train, test_X=X_chi2_test, test_y=y_test)
        total_acc.append(acc)
        total_pre.append(pre)
        total_recall.append(recall)
        total_macro_f1.append(macro_f1)
        total_micro_f1.append(micro_f1)
        del X_chi2_train, X_chi2_test, y_train, y_test
    print("======================")
    print("avg acc:", np.mean(total_acc))
    print("avg pre:", np.mean(total_pre))
    print("avg recall:", np.mean(total_recall))
    print("avg macro_f1:", np.mean(total_macro_f1))
    print("avg micro_f1:", np.mean(total_micro_f1))
    print("======================")

train_chi2_model()