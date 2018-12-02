# -*- coding: utf-8 -*-
# @Time    : 18-5-2 下午9:41
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : model_v_X.py
# @Software: PyCharm

import lightgbm as lgb
import pandas as pd
from math import log
from math import e
import numpy as np
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
        'num_leaves': 40,
        # 'max_depth': 5,
        'min_data_in_leaf': 6,
        'learning_rate': 0.020,

        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,

        'num_iterations': 300,
        # 'n_estimators': 106,

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

def train_model_mid(df_train_features, df_train_label, df_eval_features, df_eval_label):

    lgb_train = lgb.Dataset(df_train_features, label=df_train_label)

    lgb_eval = lgb.Dataset(df_eval_features, label=df_eval_label, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'rmse'},
        'num_leaves': 40,
        # 'max_depth': 5,
        'min_data_in_leaf': 6,
        'learning_rate': 0.020,

        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,

        'num_iterations': 30,
        # 'n_estimators': 106,

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
        'num_leaves': 40,
        # 'max_depth': 5,
        'min_data_in_leaf': 6,
        'learning_rate': 0.020,

        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,

        'num_iterations': 300,
        # 'n_estimators': 105,

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
    # gbm.save_model('final_model.txt')

    return gbm_final

if __name__ == "__main__":
    # load or create your dataset
    print('Load data...')
    df_train_data = pd.read_csv('/Users/yangwenjia/Code/MLTest/data/contest/1.3.2/train.1.3.2/part-00000', header=None,
                                encoding='utf-8')
    df_test_data = pd.read_csv('/Users/yangwenjia/Code/MLTest/data/contest/1.3.2/test.1.3.2/part-00000', header=None,
                               encoding='utf-8')
    df_features = df_train_data.iloc[:, 1:45].values
    df_labels = df_train_data.iloc[:, 45:50]

    df_test_features = df_test_data.iloc[:, 1:45].values

    print(df_features.shape[0], df_labels.shape[0], df_features.shape[1])
    print(df_features.shape[1], df_test_features.shape[1])
    assert df_features.shape[0] == df_labels.shape[0]
    assert df_features.shape[1] == df_test_features.shape[1]

    result = 0.0
    total = 0.0
    res = []
    mid = []
    test_mid_labels = []
    mid_train_features = []
    mid_train_labels = []
    mid_ypred = []

    for i in range(5):
        labels = []
        for j in range(len(df_labels)):
            labels.append(df_labels.values[j][i])

        gbm, y_pred, y_label, x_features = train_model(df_features, labels)

        tmp = gbm.predict(df_features, num_iteration=gbm.best_iteration)
        test = gbm.predict(df_test_features, num_iteration=gbm.best_iteration)

        mid.append(tmp)
        test_mid_labels.append(test)
        mid_train_features.append(x_features)
        mid_train_labels.append(y_label)
        mid_ypred.append(y_label)

        result += metric(y_pred.tolist(), y_label)

    """
    二级
    """
    ## 训练集合扩展特征
    X_features = []
    labels_x = np.array(mid).transpose()
    for m in range(len(df_features)) :
        x = df_features[m].tolist()
        x.extend(labels_x[m])
        X_features.append(x)

    ## 测试集合扩展特征
    X_test_features = []
    labels_test = np.array(test_mid_labels).transpose()
    for m in range(len(df_test_features)) :
        y = df_test_features[m].tolist()
        y.extend(labels_test[m])
        X_test_features.append(y)

    mid_res = 0.0
    extend_featues = pd.DataFrame(X_features).values
    extend_test_features = pd.DataFrame(X_test_features).values


    print(extend_featues.shape)
    print(extend_test_features.shape)

    for i in range(5):
        labels = []
        for j in range(len(df_labels)):
            labels.append(df_labels.values[j][i])

        # labels_train = np.array(mid_train_labels).transpose()

        train_feature_mid = []
        labels_x = np.array(mid_ypred).transpose()
        xx = mid_train_features[i]
        for m in range(len(xx)):
            x = xx[m].tolist()
            x.extend(labels_x[m])
            train_feature_mid.append(x)

        extend_mid_features = pd.DataFrame(train_feature_mid).values

        gbm_mid, y_pred, y_label_mid, x_features_mid = train_model_mid(extend_featues, labels, extend_mid_features, mid_train_labels[i])

        pred_mid = gbm_mid.predict(x_features_mid, num_iteration=gbm_mid.best_iteration)
        mid_res += metric(pred_mid, y_label_mid)

        final = train_model_final(extend_featues, labels)
        pred_final = final.predict(extend_test_features, num_iteration=final.best_iteration)
        res.append(pred_final)

        total += metric(final.predict(x_features_mid, num_iteration=final.best_iteration), y_label_mid)

    print(result / 5)
    print(mid_res / 5)
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


