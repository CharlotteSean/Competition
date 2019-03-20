# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 08:29:47 2018

@author: SY
"""

from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import warnings
import time
import pandas as pd

from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from lightgbm import plot_importance

import operator
from sklearn.model_selection import StratifiedKFold,train_test_split

warnings.filterwarnings("ignore")
nrows = None

def _read_data():
    '''这部分也是要做修改的，先放在这里'''
    train_chusai = pd.read_table('../xunfei/data/round1_iflyad_train.txt', nrows=nrows)
    train = pd.read_table('../xunfei/01_fusai/data/round2_iflyad_train.txt', nrows=nrows)
    test = pd.read_table( '../xunfei/01_fusai/data/round2_iflyad_test_feature.txt', nrows=nrows)
    data = pd.concat([train_chusai, train], axis=0, ignore_index=True)
    data = pd.concat([data, test], axis=0, ignore_index=True)
    data = data.fillna(-1).reset_index(drop=True)
    return data

data = _read_data()

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)
del data['click']

bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']

for i in bool_feature:
    data[i] = data[i].astype(int)
    
data['advert_industry_inner_0'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[1])
data['inner_slot_id_0'] = data['inner_slot_id'].fillna('_').apply(lambda x:x.split('_')[0])

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']

data['OSV'] = data['osv'].apply(lambda x:str(x).replace('_', '.').replace('Android', '').replace('android', '').replace('iOS', '').replace('iPhone OS', '').replace('iPhoneOS', '').strip(' '))
stp_lst = ['nan','unknown', 'O', 'JiangNan','\x05��5.1.1��\x06��LMY47V����\uffff\uffff\uffff\uffff',
           'Linux', 'P', 'B770']
data['OSV_0'] = data['OSV'].apply(lambda x:int(x.split('.')[0]) if x not in stp_lst else -1)
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id',]

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']
origin_cate_feat = ad_cate_feature + media_cate_feature + content_cate_feature
origin_cate_feat.append('user_tags')

add_cate_feat = ['advert_industry_inner_1','inner_slot_id_0','advert_industry_inner_0','hour','OSV','OSV_0']
cate_feat = origin_cate_feat + add_cate_feat
for i in cate_feat:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    
data['user_adid'] = list(map(lambda x,y:'_'.join([str(x),str(y)]), data['adid'],data['user_tags']))

data['user_adid'] = LabelEncoder().fit_transform(data['user_adid'])

data.drop(['time','os_name'],axis=1,inplace=True)

cate_feat += ['user_tags','user_adid']

import pickle
with open('./feature/data.plk', 'rb') as f:
    d = pickle.load(f)
data = pd.merge(data, d, on='instance_id', how='left')
data.to_csv('./feature/data.csv', index=False)
# =============================================================================
# predict = data[data.label == -1]
# predict_result = predict[['instance_id']]
# predict_result['predicted_score'] = 0
# 
# train = data[data.label != -1]
# =============================================================================

