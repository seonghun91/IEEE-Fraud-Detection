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

PREF = 'f102_'

FOLD = 5

SEED = 71


train_trans = pd.read_csv('input/train_transaction.csv')
test_trans = pd.read_csv('input/test_transaction.csv')

# timeblock으로 시간을 만드는 코드 by 현우님 
import datetime 

start_date = datetime.datetime.strptime('2017.11.30', '%Y.%m.%d')
train_trans['timeblock'] = train_trans['TransactionDT'].apply(lambda x: datetime.timedelta(seconds = x) + start_date ) 
test_trans['timeblock'] = test_trans['TransactionDT'].apply(lambda x: datetime.timedelta(seconds = x) + start_date ) 

tb = train_trans['timeblock']
train_trans.drop('timeblock', 1, inplace=True)
train_trans.insert(0, 'timeblock', tb)

tb = test_trans['timeblock']
test_trans.drop('timeblock', 1, inplace=True)
test_trans.insert(0, 'timeblock', tb)

# "가입일로부터의 시간"(D8)을 통해 "가입일"을 만드는 코드. 
def account_start_date(val):
    if np.isnan(val) :
        return np.NaN
    else:
        days=  int( str(val).split('.')[0])
        return pd.Timedelta( str(days) +' days')
    
for i in ['D1', 'D2',  'D4', 'D8','D10', 'D15']:
    train_trans['account_start_day'] = train_trans[i].apply(account_start_date)
    test_trans['account_start_day'] = test_trans[i].apply(account_start_date)

    # account_make_date 컴퓨터가 인식할 수 있도록 수치형으로 바꿔 줌. 
    train_trans['account_make_date'] = (train_trans['timeblock'] - train_trans['account_start_day']).dt.date
    test_trans['account_make_date'] = (test_trans['timeblock'] - test_trans['account_start_day']).dt.date

    train_trans['account_make_date_{}'.format(i)] = (10000 * pd.to_datetime(train_trans['account_make_date']).dt.year) + (100 * pd.to_datetime(train_trans['account_make_date']).dt.month) + (1 * pd.to_datetime(train_trans['account_make_date']).dt.day)
    test_trans['account_make_date_{}'.format(i)] = (10000 * pd.to_datetime(test_trans['account_make_date']).dt.year) + (100 * pd.to_datetime(test_trans['account_make_date']).dt.month) + (1 * pd.to_datetime(test_trans['account_make_date']).dt.day)

del train_trans['account_make_date']; del test_trans['account_make_date']
del train_trans['account_start_day']; del test_trans['account_start_day']


train_trans['date'] = pd.to_datetime(train_trans['timeblock']).dt.date
test_trans['date'] = pd.to_datetime(test_trans['timeblock']).dt.date

train_trans['year'] = train_trans['timeblock'].dt.year
train_trans['month'] = train_trans['timeblock'].dt.month
train_trans['day'] = train_trans['timeblock'].dt.day
train_trans['dayofweek'] = train_trans['timeblock'].dt.dayofweek
train_trans['hour'] = train_trans['timeblock'].dt.hour
# train_trans['minute'] = train_trans['timeblock'].dt.minute
# train_trans['second'] = train_trans['timeblock'].dt.second

test_trans['year'] = test_trans['timeblock'].dt.year
test_trans['month'] = test_trans['timeblock'].dt.month
test_trans['day'] = test_trans['timeblock'].dt.day
test_trans['dayofweek'] = test_trans['timeblock'].dt.dayofweek
test_trans['hour'] = test_trans['timeblock'].dt.hour
# test_trans['minute'] = test_trans['timeblock'].dt.minute
# test_trans['second'] = test_trans['timeblock'].dt.second



tr = train_trans[['timeblock', 'account_make_date_D1', 'account_make_date_D2',
                  'account_make_date_D4', 'account_make_date_D8', 
                  'account_make_date_D10', 'account_make_date_D15',
                  'year', 'month', 'day', 'dayofweek', 'hour']]

te = test_trans[['timeblock', 'account_make_date_D1', 'account_make_date_D2',
                  'account_make_date_D4', 'account_make_date_D8', 
                  'account_make_date_D10', 'account_make_date_D15',
                  'year', 'month', 'day', 'dayofweek', 'hour']]


tr.to_csv('f103_train.csv', index= False)
te.to_csv('f103_test.csv', index= False)