#coding=utf-8

import os
import numpy as np
import pandas as pd
from my_code.classifier import classify

class MyDoc2vecC:

    def __init__(self):
        self.cbow = 1
        self.size = 300
        self.window = 5
        self.negative = 5
        self.hs = 0
        self.sample = 0
        self.threads = 40
        self.binary = 0
        self.iter = 20
        self.min_count = 5
        self.sentence_sample = 0.1
        self.vocab_path = ""
        self.word_path = ""
        self.output_path = ""

    def train_doc2vecC(self, corpus_path, test_path):
        """

        :param corpus_path:
        content of corpus_path is formed as:

            "I am good\nYou are the best\nHow is going\n"

        :return:
        """
        # notice that "corpus_path" and "test_path" can refer to the same file, but
        # vectors in output_path are accordance with the test_path

        """parameter"""
        train = " -train " + corpus_path
        word = " -word " + self.word_path
        output_path = " -output " + self.output_path
        cbow = " -cbow " + str(self.cbow)
        size = " -size " + str(self.size)
        window = " -window " + str(self.window)
        negative = " -negative " + str(self.negative)
        hs = " -hs " + str(self.hs)
        sample = " -sample " + str(self.sample)
        thread = " -thread " + str(self.threads)
        binary = " -binary " + str(self.binary)
        iter = " -iter " + str(self.iter)
        min_count = "-min-count " + str(self.min_count)
        test = " -test " + str(test_path)
        sentence_sample = " -sentence-sample " + str(self.sentence_sample)
        save_vocab = " -save-vocab " + str(self.vocab_path)


        train_command = "time ./doc2vecc" + train + word + output_path + cbow + size + window \
                        + negative + hs + sample + thread + binary + iter + min_count + test + sentence_sample + save_vocab

        os.system("gcc doc2vecc.c -o doc2vecc -lm -pthread -O3 -march=native -funroll-loops")
        os.system(train_command)
        os.system("rm doc2vecc")

    def get_doc2vecC_vectors(self):
        rows = open(self.output_path).readlines()
        nums = len(rows) - 1
        vectors = np.zeros([nums, self.size], dtype=np.float32)
        for i in range(nums):
            values = rows[i].strip().split(" ")
            for j in range(self.size):
                vectors[i][j] = float(values[j])
        return vectors

def create_corpus():
    dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
    corpus_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/corpus.txt"
    data = pd.read_csv(dataset, sep='\t')
    contents = data['weibo']
    contents = [str(content).lower().replace("|||", " ").strip() for content in contents]
    f = open(corpus_path, 'w+')
    for content in contents:
        f.write(content + "\n")
    f.close()

# create_corpus()

def train_doc2vecC():
    corpus_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/corpus.txt"
    test_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/corpus.txt"
    dc = MyDoc2vecC()
    dc.vocab_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/vocab.txt"
    dc.word_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/word.txt"
    dc.output_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/dc_vectors.txt"
    dc.train_doc2vecC(corpus_path, test_path)

def create_training_vectors():
    dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
    data = pd.read_csv(dataset, sep='\t')
    users = data['user']

    dc_vector_file = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/doc2vecC/dc_vectors.txt"
    rows = open(dc_vector_file).readlines()
    vectors = []
    for i in range(len(rows) - 1):
        row = rows[i].strip()
        vector = [float(d) for d in row.split(" ")]
        # print(vector)
        vectors.append(vector)

    user_vector_dict = {}
    for i in range(len(users)):
        user = users[i]
        user_vector_dict[user] = vectors[i]

    label_path = "/media/iiip/Elements/数据集/user_profiling/weibo/label/age.csv"
    data = pd.read_csv(label_path, sep='\t')
    labeled_user = data['user']
    labels = data['label']
    labeled_vectors = []
    y = []
    for i in range(len(labeled_user)):
        user = labeled_user[i]
        labeled_vectors.append(user_vector_dict[user])
        y.append(int(labels[i]))
    y = np.array(y)
    labeled_vectors = np.array(labeled_vectors)
    return labeled_vectors, y




# train_doc2vecC()

from sklearn.model_selection import StratifiedKFold

def train_test():
    X_features, y = create_training_vectors()
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
