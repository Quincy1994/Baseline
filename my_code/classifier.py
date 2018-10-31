# coding=utf-8
import numpy as np
from sklearn.linear_model import LogisticRegression
from Metrics import metric_acc,score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

def classify(train_X,train_y,test_X,test_y):
    # train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
    print(np.shape(train_X))
    # clf = OneVsRestClassifier(SVC(kernel='linear'))
    # clf = svm.SVC()
    clf = LogisticRegression()
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    # print(predict_y)
    # precision = precision_score(test_y,predict_y)
    acc = metric_acc(test_y, predict_y)
    pre, recall, macro_f1, micro_f1 = score(test_y,predict_y)
    print("======================")
    print("acc:", acc)
    print("pre:", pre)
    print("recall:", recall)
    print("macro_f1:", macro_f1)
    print("micro_f1:", micro_f1)
    print("======================")
    del clf
    return acc, pre, recall, macro_f1, micro_f1