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

train_trans_Amt = pd.DataFrame(train_trans.groupby(['date','card1','TransactionAmt'])['TransactionAmt'].agg({'count'})).reset_index()
test_trans_Amt = pd.DataFrame(test_trans.groupby(['date','card1','TransactionAmt'])['TransactionAmt'].agg({'count'})).reset_index()

train_trans_Amt1 = pd.DataFrame(train_trans.groupby(['date','card3','addr1','TransactionAmt'])['TransactionAmt'].agg({'count'})).reset_index()
test_trans_Amt1 = pd.DataFrame(test_trans.groupby(['date','card3','addr1','TransactionAmt'])['TransactionAmt'].agg({'count'})).reset_index()

# Data Merge
train_trans = pd.merge(train_trans,train_trans_Amt,how='left',on=['date','card1','TransactionAmt'])
test_trans = pd.merge(test_trans,test_trans_Amt,how='left',on=['date','card1','TransactionAmt'])

# Data Merge
train_trans = pd.merge(train_trans,train_trans_Amt1,how='left',on=['date','card3','addr1','TransactionAmt'])
test_trans = pd.merge(test_trans,test_trans_Amt1,how='left',on=['date','card3','addr1','TransactionAmt'])

train_trans.to_csv('f203_train.csv', index = False)
test_trans.to_csv('f203_train.csv', index = False)

