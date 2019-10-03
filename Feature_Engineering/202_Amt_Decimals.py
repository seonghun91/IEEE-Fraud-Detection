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

PREF = 'f202_'

FOLD = 5

SEED = 71


train_trans = pd.read_csv('input/train_transaction.csv')
test_trans = pd.read_csv('input/test_transaction.csv')

train_trans['TransactionAmt_decimal_count'] = ((train_trans['TransactionAmt'] - train_trans['TransactionAmt'].astype(int))).astype(str).apply(lambda x: len(x.split('.')[1]))
test_trans['TransactionAmt_decimal_count'] = ((test_trans['TransactionAmt'] - test_trans['TransactionAmt'].astype(int))).astype(str).apply(lambda x: len(x.split('.')[1]))

train_trans['TransactionAmt_decimal'] = ((train_trans['TransactionAmt'] - train_trans['TransactionAmt'].astype(int)) * 1000).astype(int)
test_trans['TransactionAmt_decimal'] = ((test_trans['TransactionAmt'] - test_trans['TransactionAmt'].astype(int)) * 1000).astype(int)

train_trans[['TransactionAmt_decimal_count', 'TransactionAmt_decimal']].to_csv('f202_train.csv', index = False)
test_trans[['TransactionAmt_decimal_count', 'TransactionAmt_decimal']].to_csv('f202_test.csv', index = False)

