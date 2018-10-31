# coding=utf-8
import pandas as pd

# create user_label.csv
def create_label_file():
    data_path = "/media/iiip/Elements/数据集/user_profiling/weibo/label/age.txt"
    label_path = "/media/iiip/Elements/数据集/user_profiling/weibo/label/age.csv"
    rows = open(data_path).readlines()
    data = []
    for row in rows:
        row = row.strip()
        token = row.split("\t")
        user_id = token[0]
        label = token[1]
        row_dict = {}
        row_dict["user"] = user_id
        row_dict["label"] = label
        data.append(row_dict)
    data = pd.DataFrame(data)
    data = data[['user', 'label']]
    data.fillna("", inplace=True)
    data.to_csv(label_path, index=False, sep='\t')