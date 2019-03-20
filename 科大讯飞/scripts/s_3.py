# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:09:05 2018

@author: SY
"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from collections import  Counter

print('load csr ---------')
base_train_csr = sparse.load_npz('./feature/base_train_csr.npz').tocsr().astype('float32')
base_predict_csr = sparse.load_npz('./feature/base_predict_csr.npz').tocsr().astype('float32')


print('load data---------')
row_feature = ['model-', 'model+', 'model ', 'model,', 'model%', 'model_Redmi', 
               'model_s', 'model_Plus', 'osv_0', 'osv_len', 'make_new', 
               'model_Note']

usecols = ['instance_id', 'model', 'osv', 'make', 'user_tags', 'inner_slot_id', 
           'advert_industry_inner', 'advert_id', 'adid']
train_chusai = pd.read_table('../data/round1_iflyad_train.txt', usecols=usecols+['click'])
train = pd.read_table('./data/round2_iflyad_train.txt', usecols=usecols+['click'])
test  = pd.read_table('./data/round2_iflyad_test_feature.txt', usecols=usecols)
test['click'] = -1
# data = train.append(test).reset_index(drop=True)
data = pd.concat([train, test], axis=0, ignore_index=True)
data = pd.concat([train_chusai, data], axis=0, ignore_index=True)

train_y = data[data.click != -1].click.values

data['osv'] = data['osv'].astype(str)
stp_lst = ['nan','unknown', 'O', 'JiangNan','\x05��5.1.1��\x06��LMY47V����\uffff\uffff\uffff\uffff',
           'Linux', 'P', 'B770']
data['osv'] = data['osv'].apply(lambda x:x.replace('_', '.').replace('Android', '').replace('android', '').replace('iOS', '').replace('iPhone OS', '').replace('iPhoneOS', '').strip(' '))
data['osv_0'] = data['osv'].apply(lambda x:int(x.split('.')[0]) if x not in stp_lst else -1)
# =============================================================================
# data['osv_0_android'] = data['os_name'].apply(lambda x:1 if x =='android' else 0)
# data['osv_0_android'] *= data['osv_0']
# data['osv_0_ios'] = data['os_name'].apply(lambda x:1 if x =='ios' else 0)
# data['osv_0_ios'] *= data['osv_0']
# =============================================================================
# data['osv_1'] = data['osv'].apply(lambda x:int(x.split('.')[1]) if x not in stp_lst and len(x.split('.'))>1 else -1)
data['osv_len'] = data['osv'].apply(lambda x:len(x.split('.')) if x not in stp_lst else -1)
# data['osv_sum'] = data['osv'].apply(lambda x:sum([int(j)*(0.1)**i for i, j in enumerate(x.split('.'))]) if x not in stp_lst and '-' not in x else -1)

# =============================================================================
# data['advert_id'] -= 230000000
# =============================================================================

# =============================================================================
# data['f_channel'] = data['f_channel'].astype(str).apply(lambda x:x.split('_')[0])
# c = Counter(data['f_channel'].fillna('-1'))
# data['f_channel'] = LabelEncoder().fit_transform(data['f_channel'].fillna('-1').apply(lambda x:x if c[x] > 10 else '-1'))
# =============================================================================

'''交叉特征'''
data['model_adid'] = data['model'].astype(str) + data['adid'].astype(str)
data['model_adid'] = LabelEncoder().fit_transform(data['model_adid'])
c = Counter(data['model_adid'])
data['model_adid'] = data['model_adid'].apply(lambda x:x if c[x]>10 else -1)
row_feature += ['model_adid']

data['user_tags'] = data['user_tags'].astype(str).apply(lambda x:','.join([i if i!='-1' else 'o98k' for i in x.split(',')]))
data['user_tags_header'] = data.user_tags.fillna('-1').astype(str).apply(lambda x:','.join([i[:2] if len(i)>2 else '' for i in x.split(',')]))
for item in ['21', '22', '30', '99', 'ag', 'gd', 'mz', 'oc']:
    data['user_tags_header_'+item] = data.user_tags_header.apply(lambda x:sum([1 if i==item else 0 for i in x.split(',')]))
    row_feature += ['user_tags_header_'+item]
data['inner_slot_id_0'] = LabelEncoder().fit_transform(data['inner_slot_id'].fillna('_').apply(lambda x:x.split('_')[0]))
# 掉分神特 data['advert_industry_inner_0'] = LabelEncoder().fit_transform(data['advert_industry_inner'].apply(lambda x:x.split('_')[0]))
data['advert_industry_inner_1'] = LabelEncoder().fit_transform(data['advert_industry_inner'].apply(lambda x:x.split('_')[1]))
row_feature += ['inner_slot_id_0', 'advert_industry_inner_1']


data['model'] = data['model'].fillna('-1')
data['model-'] = data['model'].apply(lambda x:1 if '-' in x else 0)
data['model+'] = data['model'].apply(lambda x:1 if '+' in x else 0)
data['model '] = data['model'].apply(lambda x:1 if ' ' in x else 0)
data['model,'] = data['model'].apply(lambda x:1 if ',' in x else 0)
data['model%'] = data['model'].apply(lambda x:1 if '%' in x else 0)
data['model_Redmi'] = data['model'].apply(lambda x:1 if 'Redmi' in x or 'redmi' in x else 0)
data['model_s'] = data['model'].apply(lambda x:1 if 's' in x else 0)
data['model_Plus'] = data['model'].apply(lambda x:1 if 'Plus' in x or 'plus' in x else 0)
data['model_Note'] = data['model'].apply(lambda x:1 if 'Note' in x or 'note' in x else 0)

data['make'] = data['make'].fillna('-1').astype(str).apply(lambda x:x.lower().replace('-',' ').replace(',',' ').replace('.',' ').split(' ')[0])
# data['make'] = data['make'].apply(lambda x:x.replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', ''))
data['make_new'] = LabelEncoder().fit_transform(data['make'])
c = Counter(data['make_new'])
data['make_new'] = data['make_new'].apply(lambda x:x if c[x]>10 else -1)


train_csr = sparse.hstack(
    (sparse.csr_matrix(data[data.click!=-1][row_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(data[data.click==-1][row_feature]), base_predict_csr), 'csr').astype('float32')


xx_pre =[]
best_score = []
train_pred = pd.DataFrame(columns=['pred'], index=data[data.click != -1])
train_pred['click'] = train_y
train_pred_lst = []
test_pred_lst = []

for seed in range(5):
    print(seed)
    skf = StratifiedKFold(n_splits=8, random_state=seed+3, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
        print(index+1)
        lgb_model = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=200, reg_alpha=1, reg_lambda=1,# reg_alpha=0, reg_lambda=0.1,
            max_depth=-1, n_estimators=5000, objective='binary',
            subsample=0.7, colsample_bytree=0.3, subsample_freq=1,
            learning_rate=0.04, random_state=seed*10+index
        )
        lgb_model.fit(train_csr[train_index], train_y[train_index],
                      eval_set=[(train_csr[train_index], train_y[train_index]),
                                (train_csr[test_index], train_y[test_index])], early_stopping_rounds=50,verbose = 50)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
        print('test mean:', test_pred.mean())
        print(test_pred.max())
    #     predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
        train_pred['pred'][test_index] = lgb_model.predict_proba(train_csr[test_index])[:,1]
        xx_pre.append(test_pred)
    try:
        train_pred_lst.append(train_pred.copy())
        test_pred_lst.append(xx_pre[-5:])
    except:
        print('='*100)
print(np.mean(xx_pre))
print(log_loss(train_pred.click, train_pred.pred))
import os
os.makedirs('./lgb')
for i in range(5):
    te = data[data.click == -1]
    te = te[['instance_id']]
    s = 0
    for j in range(8):
        s += xx_pre[i*8+j][0]
    s /= 8
    te['predicted_score'] = s
    d = train_pred_lst[0].append(te).reset_index(drop=True)
    d.to_csv('./lgb/lgb_'+str(i+1)+'.csv', index=False)

'''
predict = data[data.click == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0

s = 0
for i in xx_pre:
    s += i
s = s/len(xx_pre)
predict_result['predicted_score'] = s
predict_result[['instance_id', 'predicted_score']].to_csv('./10_09_01.csv', index=False)
# 0.198962
# mean = 0.21364
'''