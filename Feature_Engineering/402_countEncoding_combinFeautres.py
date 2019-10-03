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
from tqdm import tqdm_notebook

PREF = 'f402_'

FOLD = 5

SEED = 71


train_trans = pd.read_csv('input/train_transaction.csv')
test_trans = pd.read_csv('input/test_transaction.csv')



train_trans['card1_addr1'] = train_trans['card1'].astype(str) + '_' + train_trans['addr1'].astype(str)
test_trans['card1_addr1'] = test_trans['card1'].astype(str) + '_' + test_trans['addr1'].astype(str)

train_trans['card1_addr2'] = train_trans['card1'].astype(str) + '_' + train_trans['addr2'].astype(str)
test_trans['card1_addr2'] = test_trans['card1'].astype(str) + '_' + test_trans['addr2'].astype(str)

train_trans['card1_ProductCD'] = train_trans['card1'].astype(str) + '_' + train_trans['ProductCD'].astype(str)
test_trans['card1_ProductCD'] = test_trans['card1'].astype(str) + '_' + test_trans['ProductCD'].astype(str)

train_trans['TransactionAmt_ProductCD'] = train_trans['TransactionAmt'].astype(str) + '_' + train_trans['ProductCD'].astype(str)
test_trans['TransactionAmt_ProductCD'] = test_trans['TransactionAmt'].astype(str) + '_' + test_trans['ProductCD'].astype(str)

train_trans['addr1_addr2'] = train_trans['addr1'].astype(str) + '_' + train_trans['addr2'].astype(str)
test_trans['addr1_addr2'] = test_trans['addr1'].astype(str) + '_' + test_trans['addr2'].astype(str)

categorical_variables_trans = ["card1_addr1", "card1_addr2", "card1_ProductCD",'TransactionAmt_ProductCD','addr1_addr2']

categorical_variables_idf = []

for i in tqdm_notebook(categorical_variables_trans):
    train_trans['{}_count_full'.format(i)] = train_trans[i].map(pd.concat([train_trans[i], test_trans[i]], ignore_index=True).value_counts(dropna=False))
    test_trans['{}_count_full'.format(i)] = test_trans[i].map(pd.concat([train_trans[i], test_trans[i]], ignore_index=True).value_counts(dropna=False))
    
    
features = [c for c in train.columns if c.endswith('count_full')]

train_trans[features].to_csv('f402_train.csv', index = False)
test_trans[features].to_csv('f402_test.csv', index = False)    
    