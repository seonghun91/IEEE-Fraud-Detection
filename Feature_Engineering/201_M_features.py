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

PREF = 'f201_'

FOLD = 5

SEED = 71


train_trans = pd.read_csv('input/train_transaction.csv')
test_trans = pd.read_csv('input/test_transaction.csv')


m_cols = [c for c in list(train_trans) if 'M' == c[0]]

# Use "M_cols" information
train_m = train_trans[['TransactionID'] + m_cols]
test_m = test_trans[['TransactionID'] + m_cols]

# Combination of all "M" columns
train_m['m_comb'] = ''
test_m['m_comb'] = ''
for col in m_cols:
    train_m['m_comb'] += train_m[col].astype(np.str) +'_' 
    test_m['m_comb'] += test_m[col].astype(np.str) +'_' 

# If the combination is not in the common value, replace those into "Unknown"
unique_trn_m_comb = np.unique(  train_m['m_comb'] )
unique_ts_m_comb  = np.unique(  test_m['m_comb'] )
common_m_comb = np.intersect1d( unique_trn_m_comb , unique_ts_m_comb )

train_m.loc[~train_m['m_comb'].isin(common_m_comb), 'm_comb'] = 'Unknown'
test_m.loc[~test_m['m_comb'].isin(common_m_comb), 'm_comb'] = 'Unknown'

# Sum of the null value for all "M" columns & "# of True value"
train_m['m_null_sum'] = train_m[m_cols].isnull().sum(axis=1)
train_m['m_T_sum'] = (train_m[m_cols]=='T').sum(axis=1)
test_m['m_null_sum'] = test_m[m_cols].isnull().sum(axis=1)
test_m['m_T_sum'] = (test_m[m_cols]=='T').sum(axis=1)

# Label Encoding columns related with 'M':
# 'm_comb' + m_cols
lbl = LabelEncoder()

for col in tqdm_notebook( m_cols + ['m_comb'] ):
    lbl.fit( train_m[col].fillna('Unknown') )
    train_m[col] = lbl.transform( train_m[col].fillna('Unknown')  ).astype(np.int8)
    test_m[col] = lbl.transform( test_m[col].fillna('Unknown')  ).astype(np.int8)
    
train_m = train_m[['TransactionID', 'm_comb','m_null_sum','m_T_sum']]
test_m = test_m[['TransactionID', 'm_comb','m_null_sum','m_T_sum']]

train_m.to_csv('f201_train.csv', index = False)
test_m.to_csv('f201_test.csv', index = False)
