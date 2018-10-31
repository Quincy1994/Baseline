# coding=utf-8
import pandas as pd
from my_code.traditional_baseline.BOW_TFIDF import MYTFIDF
import numpy as np

# train tfidf model
# dataset = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
# data = pd.read_csv(dataset, sep='\t')
# contents = data['weibo']
# contents = [str(content).lower().replace("|||", " ") for content in contents]
# tfidf = MYTFIDF()
# tfidf.tfidf_model_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/tfidf_model.m"
# # tfidf.train_tfidf(corpus=contents)
# tfidf_vectors = tfidf.get_tfidf_vector(corpus=contents)
# print(np.shape(tfidf_vectors))
# for i in range(tfidf.max_feature):
#     if tfidf_vectors[0][i] > 0:
#         print(tfidf_vectors[0][i])

# create training content
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
print(contents[0])
print(np.shape(y))

