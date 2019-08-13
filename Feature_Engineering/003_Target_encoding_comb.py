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

PREF = 'f004_'

FOLD = 5

SEED = 71

# 너무 많으므로 id_는 일단 배제
categorical_features = ['ProductCD',
                        'card1',
                        'card2',
                        'card3',
                        'card4',
                        'card5',
                        'card6',
                        'addr1',
                        'addr2',
                        #'P_emaildomain',
                        #'R_emaildomain',
                        'M1',
                        'M2',
                        'M3',
                        'M4',
                        'M5',
                        'M6',
                        'M7',
                        'M8',
                        'M9'
                         ]

skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)


train = pd.read_csv('input/train_transaction.csv')
test = pd.read_csv('input/test_transaction.csv')

train_id = pd.read_csv('input/train_identity.csv')
test_id = pd.read_csv('input/test_identity.csv')

train = train.merge(train_id,  how = 'left', on = 'TransactionID')
test = test.merge(test_id,  how = 'left', on = 'TransactionID')


train = train[categorical_features +['isFraud']]
test  = test[categorical_features]


col = []
cat_comb = list(combinations(categorical_features, 2))
for c1,c2 in cat_comb:
    train[f'{c1}-{c2}'] = train[c1].astype(str) + train[c2].astype(str)
    test[f'{c1}-{c2}'] = test[c1].astype(str) + test[c2].astype(str)
    col.append( f'{c1}-{c2}' )
    
    
    
# =============================================================================
# cardinality check
# =============================================================================
train['fold'] = 0
for i,(train_index, test_index) in enumerate(skf.split(train, train.isFraud)):
    train.loc[test_index, 'fold'] = i

for c in col:
    car_min = train.groupby(['fold', c]).size().min()
    print(f'{c}: {car_min}')

train.groupby(['fold', col[2]]).size()

# =============================================================================
# def
# =============================================================================
def multi(c):
    train[c+'_ta'] = 0
    for i,(train_index, test_index) in enumerate(skf.split(train, train.isFraud)):
        enc = train.iloc[train_index].groupby(c)['isFraud'].mean()
        train.set_index(c, inplace=True)
        train.iloc[test_index, -1] = enc
        train.reset_index(inplace=True)
    enc = train.groupby(c)['isFraud'].mean()
    test[c+'_ta'] = 0
    test.set_index(c, inplace=True)
    test.iloc[:,-1] = enc
    test.reset_index(inplace=True)
    
    # utils.to_feature(train[[c+'_ta']].add_prefix(PREF), 'feature/train')
    # utils.to_feature(test[[c+'_ta']].add_prefix(PREF),  'feature/test')    
    res_tr = train[[c+'_ta']].add_prefix(PREF)
    res_te = test[[c+'_ta']].add_prefix(PREF)
    
pool = Pool(8)
pool.map(multi, col)
pool.close()
    
    