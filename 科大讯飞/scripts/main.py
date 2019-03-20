# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:34:18 2018

@author: SY
"""
import warnings
warnings.filterwarnings("ignore")
import config
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from CV import CV
import os
import pickle

class Stacking(object):
    def __init__(self, data):
        self.data = data
        self.train_score_lst = []
        self.train_val_score_lst = []
        self.test_score_lst = []
        self.label_name = config.lable_name
        self.random_state = config.RANDOM_STATE
        self.c_lst = []
        
    def feed(self, arr, d=0.001):
        temp_dict = {}
        for item in self.data.columns.tolist(): temp_dict[item] = 1
        for item in arr:
            assert item in temp_dict
        # start
        '''拼接'''
        train_csr = sparse.csr_matrix((len(self.data[[self.label_name]].loc[np.logical_and(self.data[self.label_name]!=-1, self.data['val_tags']==0)]), 0))
        train_val_csr = sparse.csr_matrix((len(self.data[[self.label_name]].loc[np.logical_and(self.data[self.label_name]!=-1, self.data['val_tags']==1)]), 0))
        test_csr = sparse.csr_matrix((len(self.data[[self.label_name]].loc[self.data[self.label_name]==-1]), 0))
        _onehot_feature = []
        _cv_feature = []
        _row_feature = []
        for item in arr:
            if item not in config.type_dict: 
                _row_feature.append(item)
            elif config.type_dict[item] == 'cv':
                _cv_feature.append(item)
            elif config.type_dict[item] == 'onehot':
                _onehot_feature.append(item)
            else:
                print('name error')
                return
        for features in _onehot_feature:
            self.data[features] = LabelEncoder().fit_transform(self.data[features].astype(str))
        _train = self.data.loc[np.logical_and(self.data[self.label_name]!=-1, self.data['val_tags']==0)]
        _train_val = self.data.loc[np.logical_and(self.data[self.label_name]!=-1, self.data['val_tags']==1)]
        _test = self.data.loc[self.data[self.label_name]==-1]
        enc = OneHotEncoder()
        for feature in _onehot_feature:
            enc.fit(self.data[feature].values.reshape(-1, 1))
            train_csr = sparse.hstack((train_csr, enc.transform(_train[feature].values.reshape(-1, 1))), 'csr', 'bool')
            train_val_csr = sparse.hstack((train_val_csr, enc.transform(_train_val[feature].values.reshape(-1, 1))), 'csr', 'bool')
            test_csr = sparse.hstack((test_csr, enc.transform(_test[feature].values.reshape(-1, 1))), 'csr', 'bool')
        cv = CountVectorizer(min_df=20)
        for feature in _cv_feature:
            self.data[feature] = self.data[feature].astype(str)
            cv.fit(self.data[feature])
            train_csr = sparse.hstack((train_csr, cv.transform(_train[feature].astype(str))), 'csr', 'bool')
            train_val_csr = sparse.hstack((train_val_csr, cv.transform(_train_val[feature].astype(str))), 'csr', 'bool')
            test_csr = sparse.hstack((test_csr, cv.transform(_test[feature].astype(str))), 'csr', 'bool')
        train_csr = sparse.hstack((sparse.csr_matrix(_train[_row_feature]), train_csr), 'csr').astype('float32')
        train_val_csr = sparse.hstack((sparse.csr_matrix(_train_val[_row_feature]), train_val_csr), 'csr').astype('float32')
        test_csr = sparse.hstack((sparse.csr_matrix(_test[_row_feature]), test_csr), 'csr').astype('float32')
        
        if len(self.train_score_lst) != 0:
            for ix in range(len(self.train_score_lst)):
                train_csr = sparse.hstack((sparse.csr_matrix(np.array(self.train_score_lst[ix]).reshape(-1, 1)), train_csr), 'csr').astype('float32')
                train_val_csr = sparse.hstack((sparse.csr_matrix(np.array(self.train_val_score_lst[ix]).reshape(-1, 1)), train_val_csr), 'csr').astype('float32')
                test_csr = sparse.hstack((sparse.csr_matrix(np.array(self.test_score_lst[ix]).reshape(-1, 1)), test_csr), 'csr').astype('float32')
        '''CV,与之前的轮子直接对接'''
        lgb_params = { 'boosting_type':'gbdt', 'num_leaves':200, 
                       'reg_alpha':1, 'reg_lambda':1, 
                       'n_estimators':100000, 'objective':'binary',
                       'subsample':0.7, 'colsample_bytree':0.6, 
                       'learning_rate':0.02, 'min_child_weight':1}
        c = CV(_df=train_csr, y=_train[self.label_name].values, 
               random_state=self.random_state, is_val=False)
        c.CV(is_print=True, lgb_params=lgb_params, n_splits=5, round_cv=1)
        self.train_pred = 0
        for item in c.MS_arr:
            self.train_pred += np.array(item['pred_train'])
        self.train_pred /= len(c.MS_arr)
        self.train_score_lst.append(self.train_pred)
        self.test_score_lst.append(c.get_result(test_csr))
        self.train_val_score_lst.append(c.get_result(train_val_csr))
        self.c = c
        self.c_lst.append(c)
    def get_result(self, val):
        return self.c.get_result(val)

nrows = None
    
def _read_data():
    data = pd.read_csv('./feature/data.csv', nrows=nrows, usecols=['instance_id', 'click'])
    return data

def _get_feature(feature):
    return pd.read_csv('./feature/data.csv', nrows=nrows, usecols=feature)

if __name__ == '__main__':
    _n = 'ckpt1'
    if _n not in os.listdir('./'):
        os.makedirs('./'+_n)
    data = _read_data()
    train = data.loc[data[config.lable_name]!=-1].reset_index(drop=True)
    test = data.loc[data[config.lable_name]==-1].reset_index(drop=True)
    train['val_tags'] = 0
    test['val_tags'] = -1
    '''kfold'''
    print('k-folding...')
    kf = KFold(n_splits=8, random_state=config.RANDOM_STATE, shuffle=True)
    s_lst = []
    for ix, (train_ix, val_ix) in enumerate(kf.split(train), 1):
        p = './'+_n+'/'+str(ix)
        if p not in os.listdir('./'+_n):
            os.makedirs(p)
        train['val_tags'].loc[train_ix] = 0
        train['val_tags'].loc[val_ix] = 1
        print('Round %d'%ix)
        s = Stacking(train.append(test).reset_index(drop=True))
        for item in config.stacking_dict.items():
            print(item)
            if len(item[1]) < 5:
                print('='*50)
                continue
            '''读特征+预处理'''
            d = _get_feature(item[1]+[config.lable_name])
            d['val_tags'] = -2
            tr = d.loc[d[config.lable_name]!=-1].reset_index(drop=True)
            tr['val_tags'].loc[train_ix] = 0
            tr['val_tags'].loc[val_ix] = 1
            te = d.loc[d[config.lable_name]==-1].reset_index(drop=True)
            te['val_tags'] = -1
            s.data = tr.append(te).reset_index(drop=True)
            # feed
            s.feed(item[1])
            print(log_loss(tr[config.lable_name].loc[val_ix], s.train_val_score_lst[-1]))
        s_lst.append(s)
        with open(p+'/train_score_lst.plk', 'wb') as f:
            pickle.dump(s.train_score_lst, f)
        with open(p+'/train_val_score_lst.plk', 'wb') as f:
            pickle.dump(s.train_val_score_lst, f)
        with open(p+'/test_score_lst.plk', 'wb') as f:
            pickle.dump(s.test_score_lst, f)
        with open(p+'/train_ix.plk', 'wb') as f:
            pickle.dump(train_ix, f)
        with open(p+'/val_ix.plk', 'wb') as f:
            pickle.dump(val_ix, f)
            

