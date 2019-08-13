#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:33:05 2019

@author: jeong
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()

from itertools import combinations
from tqdm import tqdm
import sys
argv = sys.argv
import os,  gc
# import utils

PREF = 'f001_'


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

train = pd.read_csv('input/train_transaction.csv')
test = pd.read_csv('input/test_transaction.csv')

train_id = pd.read_csv('input/train_identity.csv')
test_id = pd.read_csv('input/test_identity.csv')

train = train.merge(train_id,  how = 'left', on = 'TransactionID')
test = test.merge(test_id,  how = 'left', on = 'TransactionID')

del train_id, test_id

tr_cat =  train[[var for var in train.columns if var in categorical_features]]
te_cat =  test[[var for var in test.columns if var in categorical_features]]

train.columns

tr_con = train.drop(['TransactionID','isFraud']+ categorical_features, axis = 1)
te_con = test.drop(['TransactionID']+ categorical_features, axis = 1)


train.to_csv('data/tr_all.csv', index = False)
test.to_csv('data/te_all.csv', index = False)

tr_cat.to_csv('data/tr_cat.csv', index = False)
te_cat.to_csv('data/tr_cat.csv', index = False)

tr_con.to_csv('data/tr_con.csv', index = False)
te_con.to_csv('data/tr_con.csv', index = False)



