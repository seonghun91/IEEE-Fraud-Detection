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

PREF = 'f205_'

FOLD = 5

SEED = 71


train_df = pd.read_csv('input/train_transaction.csv')
test_df = pd.read_csv('input/test_transaction.csv')

train_df_D1_ProductCD_Amt = pd.DataFrame(train_df.groupby(['date','account_make_date_D1','ProductCD'])['TransactionAmt'].agg({'count'})).reset_index()
test_df_D1_ProductCD_Amt = pd.DataFrame(test_df.groupby(['date','account_make_date_D1','ProductCD'])['TransactionAmt'].agg({'count'})).reset_index()
train_df_D1_ProductCD_Amt.columns = ['date','account_make_date_D1','ProductCD', 'ProductCD_D1_Amt_byDate']
test_df_D1_ProductCD_Amt.columns = ['date','account_make_date_D1','ProductCD','ProductCD_D1_Amt_byDate']

# Data Merge
train_df = pd.merge(train_df,train_df_D1_ProductCD_Amt,how='left',on=['date','account_make_date_D1','ProductCD'])
test_df = pd.merge(test_df,test_df_D1_ProductCD_Amt,how='left',on=['date','account_make_date_D1','ProductCD'])

total_df = pd.concat([train_df,test_df],axis=0,sort=False)
train_df = train_df.merge(total_df.groupby(['account_make_date_D1','ProductCD'])['TransactionAmt'].agg({'mean','std'}).reset_index().rename(columns={'mean':'D1_productCD_Amt_mean','std':'D1_productCD_Amt_std'}), how='left', on = ['account_make_date_D1','ProductCD'])
test_df = test_df.merge(total_df.groupby(['account_make_date_D1','ProductCD'])['TransactionAmt'].agg({'mean','std'}).reset_index().rename(columns={'mean':'D1_productCD_Amt_mean','std':'D1_productCD_Amt_std'}), how='left', on = ['account_make_date_D1','ProductCD'])

train_df['D_sum'] = train_df[['D6', 'D7', 'D8', 'D13', 'D14']].sum(axis = 1)
train_df['D_mean'] = train_df[['D6', 'D7', 'D8', 'D13', 'D14']].mean(axis = 1)
train_df['D_std'] = train_df[['D6', 'D7', 'D8', 'D13', 'D14']].std(axis = 1)
train_df['D_min'] = train_df[['D6', 'D7', 'D8', 'D13', 'D14']].min(axis = 1)
train_df['D_max'] = train_df[['D6', 'D7', 'D8', 'D13', 'D14']].max(axis = 1)
train_df['D_na_counts'] = train_df[['D6', 'D7', 'D8', 'D13', 'D14']].isna().sum(axis = 1)

test_df['D_sum'] = test_df[['D6', 'D7', 'D8', 'D13', 'D14']].sum(axis = 1)
test_df['D_mean'] = test_df[['D6', 'D7', 'D8', 'D13', 'D14']].mean(axis = 1)
test_df['D_std'] = test_df[['D6', 'D7', 'D8', 'D13', 'D14']].std(axis = 1)
test_df['D_min'] = test_df[['D6', 'D7', 'D8', 'D13', 'D14']].min(axis = 1)
test_df['D_max'] = test_df[['D6', 'D7', 'D8', 'D13', 'D14']].max(axis = 1)
test_df['D_na_counts'] = test_df[['D6', 'D7', 'D8', 'D13', 'D14']].isna().sum(axis = 1)


