# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:17:50 2018

@author: SY
"""

import warnings
warnings.filterwarnings("ignore")
'''读数据'''
import pandas as pd
import pickle
# 基本特征
basic_feature = ['adid', 'advert_id', 'advert_industry_inner', 'advert_name', 
                 'app_cate_id', 'app_id', 'campaign_id', 'carrier', 'city', 
                 'creative_has_deeplink', 'creative_height', 'creative_id', 'creative_is_download', 
                 'creative_is_voicead', 'creative_tp_dnf', 
                 'creative_type', 'creative_width', 'devtype', 'f_channel', 'inner_slot_id', 
                 'make', 'model', 'nnt', 'orderid', 'os', 'osv', 'province', 'hour', # , 'user_tags'
                 'advert_industry_inner_0', 'advert_industry_inner_1', 'OSV', 
                 'OSV_0']
data = pd.read_csv('./feature/data.csv', usecols=basic_feature+['instance_id', 'click'])
extra_feature = []

for i in range(1, 9):
    p = './ckpt1/'+str(i)
    _name = str(i)+'_'
    print(p)
    with open(p+'/train_score_lst.plk', 'rb') as f:
        train_score_lst = pickle.load(f)
    with open(p+'/train_val_score_lst.plk', 'rb') as f:
        train_val_score_lst = pickle.load(f)
    with open(p+'/test_score_lst.plk', 'rb') as f:
        test_score_lst = pickle.load(f)
    with open(p+'/train_ix.plk', 'rb') as f:
        train_ix = pickle.load(f)
    with open(p+'/val_ix.plk', 'rb') as f:
        val_ix = pickle.load(f)
    tr = data.loc[data['click']!=-1].reset_index(drop=True)
    te = data.loc[data['click']==-1].reset_index(drop=True)
    for ix in range(len(train_score_lst)):
        name_tp = _name+'pred_'+str(ix)
        extra_feature.append(name_tp)
        tr[name_tp] = 0
        tr[name_tp].loc[train_ix] = train_score_lst[ix]
        tr[name_tp].loc[val_ix] = train_val_score_lst[ix]
        te[name_tp] = test_score_lst[ix]
    data = tr.append(te).reset_index(drop=True)

data = data.drop_duplicates('instance_id')

d = pd.read_csv('./lgb/lgb_1.csv').rename(columns={'pred':'lgb_pred_1'}).drop_duplicates('instance_id')
data = pd.merge(data, d, on='instance_id', how='left')
extra_feature.append('lgb_pred_1')

d = pd.read_csv('./lgb/lgb_2.csv').rename(columns={'pred':'lgb_pred_2'}).drop_duplicates('instance_id')
data = pd.merge(data, d, on='instance_id', how='left')
extra_feature.append('lgb_pred_2')

d = pd.read_csv('./lgb/lgb_3.csv').rename(columns={'pred':'lgb_pred_3'}).drop_duplicates('instance_id')
data = pd.merge(data, d, on='instance_id', how='left')
extra_feature.append('lgb_pred_3')

d = pd.read_csv('./lgb/lgb_4.csv').rename(columns={'pred':'lgb_pred_4'}).drop_duplicates('instance_id')
data = pd.merge(data, d, on='instance_id', how='left')
extra_feature.append('lgb_pred_4')

d = pd.read_csv('./lgb/lgb_5.csv').rename(columns={'pred':'lgb_pred_5'}).drop_duplicates('instance_id')
data = pd.merge(data, d, on='instance_id', how='left')
extra_feature.append('lgb_pred_5')

del train_score_lst, train_val_score_lst, test_score_lst, train_ix, val_ix
import gc
gc.collect()
from main import Stacking
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

tr = data.loc[data['click']!=-1].reset_index(drop=True)
te = data.loc[data['click']==-1].reset_index(drop=True)
tr['val_tags'] = -2
te['val_tags'] = -1
'''kfold'''
print('k-folding...')
kf = KFold(n_splits=5, random_state=10, shuffle=True)
result_lst = []
s_lst = []
train_pred = pd.DataFrame(index=tr.index, columns=['pred'])
train_pred['click'] = tr['click']
for ix, (train_ix, val_ix) in enumerate(kf.split(tr), 1):
    print(ix)
    tr['val_tags'].loc[train_ix] = 0
    tr['val_tags'].loc[val_ix] = 1
    s = Stacking(tr.append(te))
    s.feed(basic_feature+extra_feature, d=0.05)
    print(log_loss(tr['click'].loc[val_ix], s.train_val_score_lst[-1]))
    train_pred['pred'].loc[val_ix] = s.train_val_score_lst[-1]
    result_lst.append(s.test_score_lst)

df_result = te[['instance_id']].copy()
s = 0
for i in result_lst:
    s += i
s /= len(result_lst)
df_result['predicted_score'] = s
df_result.to_csv('./result.csv', index=False)