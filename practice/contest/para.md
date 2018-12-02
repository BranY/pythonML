```python
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    'num_leaves': 100,
    'max_depth': 7,
    'min_data_in_leaf': 10,
    'learning_rate': 0.015,

    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,

    'num_iterations': 300,
    # 'n_estimators': 105,

    # 'min_gain_to_split': 0.16,
    # 'lambda_l2': 0.16,
    'bagging_freq': 5,
    'verbose': 0
}


df_features = df_train_data.iloc[:, 1:104].values
    df_labels = df_train_data.iloc[:, 104:109]

    df_test_features = df_test_data.iloc[:, 1:104].values

    print(df_features.shape[0], df_labels.shape[0], df_features.shape[1])
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
```

```
# liu yuan zhen
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    'num_leaves': 63,
    'max_depth':6,
    'min_data_in_leaf': 10,
    'learning_rate': 0.015,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'num_iterations': 1000,
    # 'min_gain_to_split': 0.16,
    # 'lambda_l2': 0.16,
    'bagging_freq': 5,
    'verbose': 0
}
```


```
67
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    'num_leaves': 73,
    'max_depth': 7,
    'min_data_in_leaf': 6,
    'learning_rate': 0.015,

    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,

    'num_iterations': 300,
    # 'n_estimators': 105,

    # 'min_gain_to_split': 0.16,
    # 'lambda_l2': 0.16,
    'bagging_freq': 5,
    'verbose': 0
}
```

