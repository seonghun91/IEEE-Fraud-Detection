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

# ['id_30','id_31','id_33','DeviceType','DeviceInfo']
train_trans['id_30_31_33_Type_Info_prev_click'] = train_trans['TransactionDT'] - train_trans.groupby(['id_30','id_31','id_33','DeviceType','DeviceInfo'])['TransactionDT'].shift(1)
test_trans['id_30_31_33_Type_Info_prev_click'] = test_trans['TransactionDT'] - test_trans.groupby(['id_30','id_31','id_33','DeviceType','DeviceInfo'])['TransactionDT'].shift(1)

train_trans['id_30_31_33_Type_Info_next_click'] = train_trans['TransactionDT'] - train_trans.groupby(['id_30','id_31','id_33','DeviceType','DeviceInfo'])['TransactionDT'].shift(-1)
test_trans['id_30_31_33_Type_Info_next_click'] = test_trans['TransactionDT'] - test_trans.groupby(['id_30','id_31','id_33','DeviceType','DeviceInfo'])['TransactionDT'].shift(-1)

train[['id_30_31_33_Type_Info_prev_click', 'id_30_31_33_Type_Info_next_click']].to_csv('f204_train.csv', index = False)
test[['id_30_31_33_Type_Info_prev_click', 'id_30_31_33_Type_Info_next_click']].to_csv('f204_test.csv', index = False)

