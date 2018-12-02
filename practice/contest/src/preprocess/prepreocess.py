# -*- coding: utf-8 -*-
# @Time    : 18-5-2 下午8:18
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : prepreocess.py
# @Software: PyCharm


import re
import pandas as pd
def read_data_from_txt(infile,outfile):
    lines = open(infile).readlines()
    dataset = []
    for line in lines:
        data = line.split('\t')
        user_id = data[0]
        features = data[1]
        features_array = features.split(',')


        if re.match(r'[\w_]+',features_array[-1]):
            label = features_array[-1].rstrip('\n').split('_')

            data_list = features_array[:-1]+label

        else:
            print("only test")
            data_list = features_array[:-1]

        dataset.append(data_list)
    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv(outfile,encoding='utf-8',header=False,index=False)


def read_data_from_csv(infile,outfile):
    lines = pd.read_csv(infile,header=None,encoding='utf-8')
    dataset = []
    for line in lines:
        data = line.split('\t')
        user_id = data[0]
        features = data[1]
        features_array = features.split(',')


        if re.match(r'[\w_]+',features_array[-1]):
            label = features_array[-1].rstrip('\n').split('_')

            data_list = features_array[:-1]+label

        else:
            print("only test")
            data_list = features_array[:-1]

        dataset.append(data_list)
    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv(outfile,encoding='utf-8',header=False,index=False)


def test_read(infile):
    data = pd.read_csv(infile,header=None,encoding='utf-8')
    print(data.iloc[:,55:60])

if __name__ == "__main__":
    test_read('../data/train_feature_with_label.csv')

    # read_data('../data/train.0.1.csv','../data/train_feature_with_label.csv')
    # read_data('../data/test.csv', '../data/test_feature.csv')