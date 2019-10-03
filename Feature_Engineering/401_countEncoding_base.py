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

PREF = 'f401_'

FOLD = 5

SEED = 71


train_trans = pd.read_csv('input/train_transaction.csv')
test_trans = pd.read_csv('input/test_transaction.csv')


categorical_variables_trans = ["ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain","R_emaildomain","P_emaildomain_bin",
                   "R_emaildomain_bin","M1","M2","M3","M4","M5","M6","M7","M8","M9",'email_null_concat']

categorical_variables_idf = ["DeviceType","DeviceInfo","id_12",
                   "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                   "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                   "id_37","id_38"]

for i in tqdm_notebook(categorical_variables_trans):
    train_trans['{}_count_full'.format(i)] = train_trans[i].map(pd.concat([train_trans[i], test_trans[i]], ignore_index=True).value_counts(dropna=False))
    test_trans['{}_count_full'.format(i)] = test_trans[i].map(pd.concat([train_trans[i], test_trans[i]], ignore_index=True).value_counts(dropna=False))
    
for i in tqdm_notebook(categorical_variables_idf):
    train_idf['{}_count_full'.format(i)] = train_idf[i].map(pd.concat([train_idf[i], test_idf[i]], ignore_index=True).value_counts(dropna=False))
    test_idf['{}_count_full'.format(i)] = test_idf[i].map(pd.concat([train_idf[i], test_idf[i]], ignore_index=True).value_counts(dropna=False))
    

features = [c for c in train.columns if c.endswith('count_full')]

train_trans[features].to_csv('f401_train.csv', index = False)
test_trans[features].to_csv('f401_test.csv', index = False)