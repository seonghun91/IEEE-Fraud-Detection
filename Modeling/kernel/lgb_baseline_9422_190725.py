"""
작성일 : 19/07/24
pb score : 0.9420
특징 : card_1 value_count feature 제작하여 반영

아래 베이스라인 코드를 참고하여 수정한 코드
(https://www.kaggle.com/artgor/eda-and-models)

"""


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import gc
import time
from tqdm import tqdm

import os
import time
import datetime
import json
import gc
from numba import jit

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics

from itertools import product

import altair as alt
from altair.vega import v5
from IPython.display import HTML

%%time
train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')

train_identity = pd.read_csv('../input/train_identity.csv')
test_identity = pd.read_csv('../input/test_identity.csv')
sub = pd.read_csv('../input/sample_submission.csv')

train = train_transaction.merge(train_identity, how='left', on = 'TransactionID')
test = test_transaction.merge(test_identity, how='left', on = 'TransactionID')

print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

tr2 = train
for i in range(1, 10) :
    col = 'id_0'+str(i)
    if train[col].nunique() < 100 :
        a = list(set(train[col].unique()) - set(test[col].unique()))
        a = [x for x in a if str(x) != 'nan']
        tr2 = tr2.loc[~tr2[col].isin(a)]
print(train.shape)
print(tr2.shape)

for i in range(11, 39) :
    col = 'id_'+str(i)
    if train[col].nunique() < 100 :
        a = list(set(train[col].unique()) - set(test[col].unique()))
        a = [x for x in a if str(x) != 'nan']
        tr2 = tr2.loc[~tr2[col].isin(a)]
print(tr2.shape)


for i in range(1, 340) :
    col = 'V'+str(i)
    if train[col].nunique() < 100 :
        a = list(set(train[col].unique()) - set(test[col].unique()))
        a = [x for x in a if str(x) != 'nan']
        tr2 = tr2.loc[~tr2[col].isin(a)]
print(tr2.shape)

for i in range(1, 10) :
    col = 'M'+str(i)
    if train[col].nunique() < 100 :
        a = list(set(train[col].unique()) - set(test[col].unique()))
        a = [x for x in a if str(x) != 'nan']
        tr2 = tr2.loc[~tr2[col].isin(a)]
print(tr2.shape)

for i in range(1, 16) :
    col = 'D'+str(i)
    if train[col].nunique() < 100 :
        a = list(set(train[col].unique()) - set(test[col].unique()))
        a = [x for x in a if str(x) != 'nan']
        tr2 = tr2.loc[~tr2[col].isin(a)]
for i in range(1, 15) :
    col = 'C'+str(i)
    if train[col].nunique() < 100 :
        a = list(set(train[col].unique()) - set(test[col].unique()))
        a = [x for x in a if str(x) != 'nan']
        tr2 = tr2.loc[~tr2[col].isin(a)]
print(tr2.shape)
train = tr2

train['hour'] = round(train['TransactionDT'] / 60 / 60 ) % 24
test['hour'] = round(test['TransactionDT'] / 60 / 60 ) % 24

# train['hour'] = train['hour'].astype("object")
# test['hour'] = test['hour'].astype("object")
##\train['week'] = round(train['TransactionDT'] / 60 / 60 /24) // 7
#test['week'] = round(test['TransactionDT'] / 60 / 60 /24) // 7

#train['week'] = train['week'].astype("object")
#test['week'] = test['week'].astype("object")
# feature engineering
train['ProductCD_card4'] = train['ProductCD'] + train['card4']
test['ProductCD_card4'] = test['ProductCD'] + test['card4']

train['ProductCD_card6'] = train['ProductCD'] + train['card6']
test['ProductCD_card6'] = test['ProductCD'] + test['card6']

train['ProductCD_card4_card6'] = train['ProductCD'] + train['card4'] + train['card6']
test['ProductCD_card4_card6'] = test['ProductCD'] + test['card4']  + test['card6']

train['day'] = round(train['TransactionDT'] / 60 / 60 /24 )
test['day'] = round(test['TransactionDT'] / 60 / 60 /24 ) 

test.loc[test['day'] > 365 , 'day'] = test['day'] - 365

train['feat1'] = 'B'
test['feat1'] = 'B'

train.loc[train['day'] < 25 , 'feat1'] = 'A'
test.loc[test['day'] < 25 , 'feat1'] = 'A'

train['feat1'] = train['feat1'].astype('object')
test['feat1'] = test['feat1'].astype('object')

train['feat1_ProductCD'] = train['feat1'] + train['ProductCD']
test['feat1_ProductCD'] = test['feat1'] + test['ProductCD']

train['feat1_card4'] = train['feat1'] + train['card4']
test['feat1_card4'] = test['feat1'] + test['card4']

train['feat1_card6'] = train['feat1'] + train['card6']
test['feat1_card6'] = test['feat1'] + test['card6']

train['feat1_card4_ProductCD'] = train['feat1'] + train['card4'] + train['ProductCD']
test['feat1_card4_ProductCD'] = test['feat1'] + test['card4'] + test['ProductCD']

train['card3_150'] = 0
train.loc[train['card3'] == 150, 'card3_150'] = 1

test['card3_150'] = 0
test.loc[test['card3'] == 150, 'card3_150'] = 1

train['card3_185'] = 0
train.loc[train['card3'] == 185, 'card3_185'] = 1

test['card3_185'] = 0
test.loc[test['card3'] == 185, 'card3_185'] = 1


train['card1_card2'] = train['card1'].astype('str') + train['card2'].astype('str') 
test['card1_card2'] = test['card1'].astype('str') + test['card2'].astype('str') 

train['card1_P_emaildomain'] = train['card1'].astype('str') + train['P_emaildomain'].astype('str') 
test['card1_P_emaildomain'] = test['card1'].astype('str') + test['P_emaildomain'].astype('str') 

d1 = train.groupby(['card1']).size().reset_index()
d1.columns = ['card1','card1_cnt']

train = pd.merge(train, d1, how = 'left', on = 'card1')
test = pd.merge(test, d1, how = 'left', on = 'card1')
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
one_value_cols == one_value_cols_test

many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]
big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))
cols_to_drop.remove('isFraud')
len(cols_to_drop)

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)
cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3',
           'ProductCD_card4', 'ProductCD_card6', 'ProductCD_card4_card6', 'day', 'hour', 'feat1', 'feat1_ProductCD', 'feat1_card4', 'feat1_card6','card1_card2','card1_P_emaildomain',
       'feat1_card4_ProductCD']

for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))   
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
del train
test = test[["TransactionDT", 'TransactionID']]
gc.collect()

n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)
folds = KFold(n_splits=5)
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True
def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual'):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                        'catboost_metric_name': 'AUC',
                        'sklearn_scoring_function': metrics.roc_auc_score},
                    }
    
    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict_proba(X_test)
        
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        if averaging == 'usual':
            
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            
            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':
                                  
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
                                  
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols
        
    return result_dict
params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9
         }
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual')


test = test.sort_values('TransactionDT')
test['prediction'] = result_dict_lgb['prediction']
sub['isFraud'] = pd.merge(sub, test, on='TransactionID')['prediction']
sub.to_csv('submission.csv', index=False)
sub.head()

