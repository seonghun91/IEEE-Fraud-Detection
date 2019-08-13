#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:22:31 2019

@author: jeong
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm

PREF = 'f002_'
FOLD = 5
SEED = 59


categorical_features = ['ProductCD',
                        'card1',
                        'card2',
                        'card3',
                        'card4',
                        'card5',
                        'card6',
                        'addr1',
                        'addr2',
                        'P_emaildomain',
                        'R_emaildomain',
                        'M1',
                        'M2',
                        'M3',
                        'M4',
                        'M5',
                        'M6',
                        'M7',
                        'M8',
                        'M9',
                        'DeviceType',
                        'DeviceInfo',
                        'id_12',
                        'id_13',
                        'id_14',
                        'id_15',
                        'id_16',
                        'id_17',
                        'id_18',
                        'id_19',
                        'id_20',
                        'id_21',
                        'id_22',
                        'id_23',
                        'id_24',
                        'id_25',
                        'id_26',
                        'id_27',
                        'id_28',
                        'id_29',
                        'id_30',
                        'id_31',
                        'id_32',
                        'id_33',
                        'id_34',
                        'id_35',
                        'id_36',
                        'id_37',
                        'id_38',
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

usecols = []
for c in tqdm(categorical_features):
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
    
    usecols.append(c+'_ta')




# =============================================================================
# cardinality check
# =============================================================================

train['fold'] = 0
for i,(train_index, test_index) in enumerate(skf.split(train, train.isFraud)):
    train.loc[test_index, 'fold'] = i

for c in categorical_features:
    car_min = train.groupby(['fold', c]).size().min()
    print(f'{c}: {car_min}')


# output
# utils.to_feature(train[usecols].add_prefix(PREF), '../feature/train')
# utils.to_feature(test[usecols].add_prefix(PREF),  '../feature/test')


