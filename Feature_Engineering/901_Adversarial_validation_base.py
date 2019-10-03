#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:29:11 2019

@author: jeong
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

PREF = 'f901_'

FOLD = 5

SEED = 71


train_df = pd.read_csv('input/train_transaction.csv')
test_df = pd.read_csv('input/test_transaction.csv')

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

train = train_df.copy()
test = test_df.copy()

from sklearn import model_selection, preprocessing, metrics

train['target'] = 0
test['target'] = 1

train_test = pd.concat([train, test], axis =0)
target = train_test['target'].values

train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()

train = lgb.Dataset(train[features], label=train_y)
test = lgb.Dataset(test[features], label=test_y)

# 문제점
# 파라미터에 따라서 아래와 결과가 달라짐. 
params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }



num_round = 50
clf = lgb.train(params, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')

feature_imp.sort_values(by='Value',ascending=False)