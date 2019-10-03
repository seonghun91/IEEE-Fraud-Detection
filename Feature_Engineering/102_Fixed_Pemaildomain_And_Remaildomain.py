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

# Email
def email_categorical_expression(emails):
    """
    Get the type of email
    (1) Both "P_emaildomain" & "R_emaildomain" are None
    (2) "P_emaildomain" is None, but "R_emaildomain" isn't None
    (3) "P_emaildomain" isn't None, but "R_emaildomain" is None
    (4) Both "P_emaildomain" & "R_emaildomain" aren't None
    """
    P_emaildomain, R_emaildomain = emails

    if type(P_emaildomain) ==  float:
        if type(R_emaildomain) == float:
            email_type = 1
        else:
            email_type = 2
    else:
        if type(R_emaildomain) == float:
            email_type = 3
        else:
            email_type = 4
    return email_type    
    
def email_null_concat(emails):
    """
    Get the row-wise concat of email_address
    """
    temp = emails.isnull().astype(np.int8)
    label= ''
    for col in ['P_emaildomain','R_emaildomain']:
        label += str(temp[col] ) +'_'
    return label

# Implement
train_trans['email_type'] = train_trans[['P_emaildomain', 'R_emaildomain']].progress_apply(lambda x : email_categorical_expression(x) , axis=1)
train_trans['email_null_concat'] = train_trans[['P_emaildomain', 'R_emaildomain']].progress_apply(lambda x : email_null_concat(x) , axis=1)

test_trans['email_type'] = test_trans[['P_emaildomain', 'R_emaildomain']].progress_apply(lambda x : email_categorical_expression(x) , axis=1)
test_trans['email_null_concat'] = test_trans[['P_emaildomain', 'R_emaildomain']].progress_apply(lambda x : email_null_concat(x) , axis=1)



# email preprocessing 
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']

emaildomain = ['P_emaildomain', 'R_emaildomain']
for c in emaildomain:
    train_trans[c + '_bin'] = train_trans[c].map(emails)
    test_trans[c + '_bin'] = test_trans[c].map(emails)
    
    train_trans[c + '_suffix'] = train_trans[c].map(lambda x: str(x).split('.')[-1])
    test_trans[c + '_suffix'] = test_trans[c].map(lambda x: str(x).split('.')[-1])
    
    train_trans[c + '_suffix'] = train_trans[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test_trans[c + '_suffix'] = test_trans[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

tr = train_trans[['email_type', 'email_null_concat', 'P_emaildomain_bin',
                  'P_emaildomain_suffix','R_emaildomain_bin', 'R_emaildomain_suffix' ]]

te = test_trans[['email_type', 'email_null_concat', 'P_emaildomain_bin',
                  'P_emaildomain_suffix','R_emaildomain_bin', 'R_emaildomain_suffix' ]]


tr.to_csv('f102_train.csv', index= False)
te.to_csv('f102_test.csv', index= False)

