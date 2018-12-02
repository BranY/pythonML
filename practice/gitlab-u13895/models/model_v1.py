# -*- coding: utf-8 -*-
# @Time    : 18-5-2 下午9:41
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : model_v1.py
# @Software: PyCharm

import lightgbm as lgb
import pandas as pd
from math import log
from math import e
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def metric(y_pred, y_true):
    length = len(y_pred)
    assert length == len(y_true)

    sum = 0
    for i in range(length):
        value = round(y_pred[i], 3)
        y_pred_log = log(value + 1)
        y_true_log = log(y_true[i] + 1)
        sum += pow((y_pred_log - y_true_log), 2)
    return sum / length


def train_model(features, labels):
    df_train_features, df_eval_features, df_train_label, df_eval_label = \
        train_test_split(features, labels, test_size=0.2)
    lgb_train = lgb.Dataset(df_train_features, label=df_train_label)

    lgb_eval = lgb.Dataset(df_eval_features, label=df_eval_label, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'rmse'},
        'num_leaves': 170,
        'min_data_in_leaf': 10,
        'learning_rate': 0.015,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'num_iterations': 150,

        # 'min_gain_to_split': 0.16,
        # 'lambda_l2': 0.16,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=80,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20)

    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    # eval
    y_pred = gbm.predict(df_eval_features, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(df_eval_label, y_pred) ** 0.5)
    return gbm, y_pred, df_eval_label, df_eval_features


def train_model_final(features, labels):
    lgb_train = lgb.Dataset(features, label=labels)

    # lgb_eval = lgb.Dataset(df_eval_features, label=df_eval_label, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'rmse'},
        'num_leaves': 170,
        'min_data_in_leaf': 10,
        'learning_rate': 0.015,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'num_iterations': 150,

        # 'min_gain_to_split': 0.16,
        # 'lambda_l2': 0.16,
        'bagging_freq': 5,
        'verbose': 0
    }

    # train
    gbm_final = lgb.train(params,
                    lgb_train,
                    num_boost_round=90
                          )

    return gbm_final

if __name__ == "__main__":
    # load or create your dataset
    print('Load data...')
    df_train_data = pd.read_csv('../data/train.1.2.2.csv', header=None, encoding='utf-8')
    df_test_data = pd.read_csv('../data/test.1.2.2.csv', header=None, encoding='utf-8')
    df_features = df_train_data.iloc[:, 1:104].values


    df_labels = df_train_data.iloc[:, 104:109]

    df_test_features = df_test_data.iloc[:, 1:104].values
    print(df_features.shape[0],df_labels.shape[0],df_features.shape[1])
    assert df_features.shape[0] == df_labels.shape[0]
    assert df_features.shape[1] == df_test_features.shape[1]

    result = 0.0
    total = 0.0
    res = []

    for i in range(5):
        labels = []
        for j in range(len(df_labels)):
            labels.append(df_labels.values[j][i])

        X_features = df_train_data.drop([0, 104+i], axis=1, inplace=False).values

        gbm, y_pred, y_label, x_features = train_model(X_features, labels)

        pred = gbm.predict(df_test_features, num_iteration=gbm.best_iteration)

        gbm_final = train_model_final(X_features, labels)

        pred_final = gbm_final.predict(df_test_features, num_iteration=gbm_final.best_iteration)

        res.append(pred_final)
        result += metric(y_pred.tolist(), y_label)
        total += metric(gbm_final.predict(x_features, num_iteration=gbm_final.best_iteration), y_label)

    print(result / 5)

    print(total / 5)

    final = []
    index = 0
    for i in range(len(res[0])):

        str_tmp = []

        str_tmp.append(df_test_data[0][i])
        for index in range(5):
            str_tmp.append(round(res[index][i], 3))

        final.append(str_tmp)



    save = pd.DataFrame(final)

    save.to_csv("../data/result.0.0.csv", index=False, header=False)
    print()


