# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:32:03 2018

@author: SY
"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import lightgbm as lgb
import time
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from collections import  Counter
import pickle
import datetime
print('load data---------')
train_chusai = pd.read_table('./data/round1_iflyad_train.txt')
train = pd.read_table('./01_fusai/data/round2_iflyad_train.txt')
test = pd.read_table( './01_fusai/data/round2_iflyad_test_feature.txt')
test['click'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
data = pd.concat([train_chusai, data], axis=0, ignore_index=True)
del train, train_chusai, test


data['osv'] = data['osv'].astype(str)
stp_lst = ['nan','unknown', 'O', 'JiangNan','\x05��5.1.1��\x06��LMY47V����\uffff\uffff\uffff\uffff','Linux','P','B770']
data['osv_0'] = data['osv'].apply(lambda x:int(x.replace('_', '.').replace('Android', '').replace('android', '').replace('iOS', '').replace('iPhone OS', '').replace('iPhoneOS', '').strip(' ').split('.')[0]) if x not in stp_lst else -1)
data['osv_len'] = data['osv'].apply(lambda x:len(x.replace('_', '.').replace('Android', '').replace('android', '').replace('iOS', '').replace('iPhone OS', '').replace('iPhoneOS', '').strip(' ').split('.')) if x not in stp_lst else -1)

data['model'] = data['model'].fillna('-1')
data['model-'] = data['model'].apply(lambda x:1 if '-' in x else 0)
data['model+'] = data['model'].apply(lambda x:1 if '+' in x else 0)
data['model '] = data['model'].apply(lambda x:1 if ' ' in x else 0)
data['model,'] = data['model'].apply(lambda x:1 if ',' in x else 0)
data['model%'] = data['model'].apply(lambda x:1 if '%' in x else 0)
data['model_Redmi'] = data['model'].apply(lambda x:1 if 'Redmi' in x or 'redmi' in x else 0)
data['model_s'] = data['model'].apply(lambda x:1 if 's' in x else 0)
data['model_Plus'] = data['model'].apply(lambda x:1 if 'Plus' in x or 'plus' in x else 0)

data['make'] = data['make'].fillna('-1').astype(str).apply(lambda x:x.lower().replace('-',' ').replace(',',' ').replace('.',' ').split(' ')[0])
# data['make'] = data['make'].apply(lambda x:x.replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', ''))
data['make_new'] = LabelEncoder().fit_transform(data['make'])
c = Counter(data['make_new'])
data['make_new'] = data['make_new'].apply(lambda x:x if c[x]>10 else -1)

row_feature = ['model-', 'model+', 'model ', 'model,', 'model%', 'model_Redmi', 
               'model_s', 'model_Plus', 'osv_0', 'osv_len', 'make_new']

data = data.fillna(-1)
bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
for i in bool_feature:
    data[i] = data[i].astype(int)
    
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']
origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

data.drop(['os_name'],axis=1,inplace=True)
cate_feature = origin_cate_list
num_feature = ['creative_width', 'creative_height', 'hour','area']

feature = cate_feature + num_feature
print(len(feature), feature)

data['area'] = data['creative_height'] * data['creative_width']

# =============================================================================
# 
# =============================================================================


data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

#认为构建uid
data['uid'] = data[['city','os','make','carrier','model']].apply(lambda x: '_'.join([str(i) for i in x]),axis=1)

data['uid'] = data['uid'].map(dict(zip(data['uid'].unique(),range(0,data['uid'].nunique()))))

#每个用户的数量
tmp =data.groupby('uid')['uid'].size().to_frame('uid_cnt').reset_index()
data_temp = pd.merge(data[['uid']],tmp,on='uid',how='left').fillna(-1)
lst = tmp.columns.tolist()
lst.pop(lst.index('uid'))
data = data.join(data_temp[lst])
# data = pd.merge(data,tmp,on='uid',how='left')
row_feature.append('uid_cnt')
#广告，广告主，广告主行业，创意等的种类数
cols = ['creative_tp_dnf','creative_type','app_id','inner_slot_id','adid','advert_id','advert_industry_inner_1',\
           'app_cate_id','day']
tmp = data.groupby('uid')[cols].agg({'nunique'})
tmp.columns = ['uid'+'_'+'_'.join(i) for i in tmp.columns]
tmp.drop(tmp.columns[tmp.nunique()==1],axis=1,inplace=True)
row_feature += list(tmp.columns)
tmp = tmp.reset_index()
data_temp = pd.merge(data[['uid']],tmp,on='uid',how='left').fillna(-1)
lst = tmp.columns.tolist()
lst.pop(lst.index('uid'))
data = data.join(data_temp[lst])
# data =pd.merge(data,tmp,on='uid',how='left')


#点击时间间隔统计特征
tmp= data.sort_values(['uid','time'])
tmp['ts_diff'] = tmp.groupby(['uid'])['time'].diff(-1).apply(np.abs)
tmp = tmp[['uid','ts_diff']].dropna()

ts = tmp.groupby(['uid'],as_index=False)['ts_diff'].agg({
    'ts_diff_mean':lambda x: np.mean,
    'ts_diff_std':lambda x : np.std,
    'ts_diff_min':lambda x: np.min,
    'ts_diff_max':lambda x: np.max
})
data_temp = pd.merge(data[['uid']],ts,on='uid',how='left').fillna(-1)
lst = ts.columns.tolist()
lst.pop(lst.index('uid'))
data = data.join(data_temp[lst])
row_feature += ['ts_diff_mean','ts_diff_min','ts_diff_max','ts_diff_std']

#用户当天点击的总次数及占比
tmp = data.groupby(['uid','day'])['model'].size().to_frame('uid_day_cnt').reset_index()
data_temp = pd.merge(data[['uid']],tmp,on='uid',how='left').fillna(-1)
lst = tmp.columns.tolist()
lst.pop(lst.index('uid'))
lst.pop(lst.index('day'))
data = data.join(data_temp[lst])
# data = pd.merge(data,tmp,on=['uid','day'],how='left')
row_feature.append('uid_day_cnt')

#用户当前广告,创意等点击次数及占比
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','adid','advert_id','advert_industry_inner_1',\
           'app_cate_id']:
    tmp1 = data.groupby(['uid'])[col].value_counts().to_frame('uid_'+col+'_cnt').reset_index()
    tmp2 = data.groupby(['uid'])[col].value_counts(normalize=True).to_frame('uid_'+col+'_cnt_ratio').reset_index()
    data_temp = pd.merge(data[['uid',col]],tmp1,on=['uid',col],how='left')
    lst = tmp1.columns.tolist()
    lst.pop(lst.index('uid'))
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    
    data_temp = pd.merge(data[['uid',col]],tmp2,on=['uid',col],how='left')
    lst = tmp2.columns.tolist()
    lst.pop(lst.index('uid'))
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,tmp1,on=['uid',col],how='left')
    # data = pd.merge(data,tmp2,on=['uid',col],how='left')
    row_feature +=['uid_'+col+'_cnt','uid_'+col+'_cnt_ratio']
            
#每个var的时间的平均值，方差等
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make','advert_id','advert_industry_inner_1',\
           'app_cate_id']:
    tmp = data.groupby(col)['hour'].agg({col+'_hour_mean':'mean',col+'_hour_std':'std'})
    data_temp = pd.merge(data[[col]],tmp,on=col,how='left').fillna(-1)
    lst = data_temp.columns.tolist()
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,tmp,on=col,how='left').fillna(-1)
    row_feature += [col+'_hour_mean',col+'_hour_std']
print(1)
#每个var的随时间间隔的统计特征
for col in ['creative_tp_dnf','creative_type','inner_slot_id','adid','make','advert_id','advert_industry_inner_1']:
    tmp = dict(list(data.groupby(col)['time']))
    d ={}
    for i in set(tmp.keys()):
        ts = tmp[i].sort_values().diff().dropna()
        d[i] = [np.mean(ts),np.max(ts),np.std(ts)]
    df = pd.DataFrame(d).T
    df.columns =[col+'_ts_mean',col+'_ts_max',col+'_ts_std']
    df = df.reset_index()
    df = df.rename(columns={'index':col})
    data_temp = pd.merge(data[[col]],df,on=col,how='left').fillna(-1)
    lst = df.columns.tolist()
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,df,on=col,how='left').fillna(-1)
    row_feature += [col+'_ts_mean',col+'_ts_max',col+'_ts_std']
# =============================================================================
#     
# =============================================================================
# # #计算每天数量的方差
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make','advert_id','advert_industry_inner_1',\
           'app_cate_id']:
    tmp = data.groupby([col,'day'])['day'].size().unstack().fillna(0).std(axis=1).to_frame(col+'_day_std').reset_index()
    row_feature.append(col+'_day_std')
    data_temp = pd.merge(data[[col]],tmp,on=col,how='left').fillna(-1)
    lst = tmp.columns.tolist()
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,tmp.reset_index(),on=col,how='left')
    
#被展示广告当天的竞争性
day_adid_ratio = data.groupby('day')['adid'].value_counts(normalize=True).to_frame('adid_day_ratio').fillna(0)
data = pd.merge(data,day_adid_ratio.reset_index(),on=['day','adid'],how='left')
row_feature.append('adid_day_ratio')
print(2)
#被展现创意，广告主行业等竞争性
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_industry_inner','advert_name','osv','model','make']:
    day_ratio = data.groupby('day')[col].value_counts(normalize=True).to_frame(col+'_day_ratio').fillna(0).reset_index()
    data_temp = pd.merge(data[['day',col]],day_ratio,on=['day',col],how='left').fillna(-1)
    lst = day_ratio.columns.tolist()
    lst.pop(lst.index('day'))
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,day_ratio.reset_index(),on=['day',col],how='left')
    row_feature.append(col+'_day_ratio')


# #长宽比
data['w_h_ratio'] =data['creative_width'] / data['creative_height']
row_feature.append('w_h_ratio')


#长宽，面积的平均
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make']:
    tmp = data.groupby(col)['area'].agg({col+'_area_mean':'mean',col+'_area_std':'std'}).fillna(-1).reset_index()
    data_temp = pd.merge(data[[col]],tmp,on=col,how='left').fillna(-1)
    lst = tmp.columns.tolist()
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,tmp,on=col,how='left')
    row_feature += [col+'_area_mean',col+'_area_std']
print(3)

var ='creative_width'
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make']:
    tmp = data.groupby(col)[var].agg({col+'_width_mean':'mean',col+'_width_std':'std'}).fillna(-1).reset_index()
    data_temp = pd.merge(data[[col]],tmp,on=col,how='left').fillna(-1)
    lst = tmp.columns.tolist()
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,tmp,on=col,how='left')
    row_feature += [col+'_width_mean',col+'_width_std']

var ='creative_height'
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make']:
    tmp = data.groupby(col)[var].agg({col+'_height_mean':'mean',col+'_height_std':'std'}).fillna(-1).reset_index()
    data_temp = pd.merge(data[[col]],tmp,on=col,how='left').fillna(-1)
    lst = tmp.columns.tolist()
    lst.pop(lst.index(col))
    data = data.join(data_temp[lst])
    # data = pd.merge(data,tmp,on=col,how='left')
    row_feature += [col+'_height_mean',col+'_height_std']

import os
os.makedirs('./feature')
with open('./feature/data.plk', 'wb') as f:
    pickle.dump(data, f)