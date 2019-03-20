# -*- coding: utf-8 -*-
"""

@author: taopeng
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
import gc
import time


#导入基本特征和原始数据
print('load csr ---------')
base_train_csr = sparse.load_npz('./feature/base_train_csr.npz').tocsr().astype('float32')
base_predict_csr = sparse.load_npz('./feature/base_predict_csr.npz').tocsr().astype('float32')

print('load data---------')

train_chusai = pd.read_table('../data/round1_iflyad_train.txt')
train = pd.read_table('./data/round2_iflyad_train.txt')
test = pd.read_table( './data/round2_iflyad_test_feature.txt')
test['click'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
data = pd.concat([train_chusai, data], axis=0, ignore_index=True)
print(len(data))

train_y = data[data.click != -1].click.values

#对数据进行清洗并构造特征
data['osv'] = data['osv'].astype(str)
stp_lst = ['nan','unknown', 'O', 'JiangNan','\x05��5.1.1��\x06��LMY47V����\uffff\uffff\uffff\uffff',
           'Linux', 'P', 'B770']
data['osv'] = data['osv'].apply(lambda x:x.replace('_', '.').replace('Android', '').replace('android', '').replace('iOS', '').replace('iPhone OS', '').replace('iPhoneOS', '').strip(' '))
data['osv_0'] = data['osv'].apply(lambda x:int(x.split('.')[0]) if x not in stp_lst else -1)
data['osv_len'] = data['osv'].apply(lambda x:len(x.split('.')) if x not in stp_lst else -1)

row_feature = ['model-', 'model+', 'model ', 'model,', 'model%', 'model_Redmi', 
               'model_s', 'model_Plus', 'osv_0', 'osv_len', 'make_new', 
               'model_Note']
#交叉特征
data['model_adid'] = data['model'].astype(str) + data['adid'].astype(str)
data['model_adid'] = LabelEncoder().fit_transform(data['model_adid'])
c = Counter(data['model_adid'])
data['model_adid'] = data['model_adid'].apply(lambda x:x if c[x]>10 else -1)
row_feature += ['model_adid']

#用户特征细节
data['user_tags'] = data['user_tags'].astype(str).apply(lambda x:','.join([i if i!='-1' else 'o98k' for i in x.split(',')]))
data['user_tags_header'] = data.user_tags.fillna('-1').astype(str).apply(lambda x:','.join([i[:2] if len(i)>2 else '' for i in x.split(',')]))
for item in ['21', '22', '30', '99', 'ag', 'gd', 'mz', 'oc']:
    data['user_tags_header_'+item] = data.user_tags_header.apply(lambda x:sum([1 if i==item else 0 for i in x.split(',')]))
    row_feature += ['user_tags_header_'+item]
data['inner_slot_id_0'] = LabelEncoder().fit_transform(data['inner_slot_id'].fillna('_').apply(lambda x:x.split('_')[0]))
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
data['make_new'] = LabelEncoder().fit_transform(data['make'])
c = Counter(data['make_new'])
data['make_new'] = data['make_new'].apply(lambda x:x if c[x]>10 else -1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
# #计算每天数量的方差
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make','advert_id','advert_industry_inner_1',\
           'app_cate_id']:
    data[col] = data[col].fillna('-1')
    tmp = data.groupby([col,'day'])['day'].size().unstack().fillna(0).std(axis=1).to_frame(col+'_day_std')
    row_feature.append(col+'_day_std')
    data = pd.merge(data,tmp.reset_index(),on=col,how='left')

#被展示广告当天的竞争性
day_adid_ratio = data.groupby('day')['adid'].value_counts(normalize=True).to_frame('adid_day_ratio').fillna(0)
data = pd.merge(data,day_adid_ratio.reset_index(),on=['day','adid'],how='left')
row_feature.append('adid_day_ratio')

# #长宽比
data['creative_width'] = data['creative_width'].fillna(-1)
data['creative_height'] =data['creative_height'].fillna(-1)
data['w_h_ratio'] =data['creative_width'] / data['creative_height']
row_feature.append('w_h_ratio')

#每个变量的长宽，面积的平均值和标准差
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make']:
    data[col] = data[col].fillna('-1')
    tmp = data.groupby(col)['area'].agg({col+'_area_mean':'mean',col+'_area_std':'std'}).fillna(-1)
    data = pd.merge(data,tmp,on=col,how='left')
    row_feature += [col+'_area_mean',col+'_area_std']

var ='creative_width'
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make']:
    tmp = data.groupby(col)[var].agg({col+'_width_mean':'mean',col+'_width_std':'std'}).fillna(-1)
    data = pd.merge(data,tmp,on=col,how='left')
    row_feature += [col+'_width_mean',col+'_width_std']

var ='creative_height'
for col in ['creative_tp_dnf','creative_type','app_id','inner_slot_id','advert_name','adid','osv','model','make']:
    tmp = data.groupby(col)[var].agg({col+'_height_mean':'mean',col+'_height_std':'std'}).fillna(-1)
    data = pd.merge(data,tmp,on=col,how='left')
    row_feature += [col+'_height_mean',col+'_height_std']



del train,test,train_chusai,tmp
gc.collect()
iid = data[data['click']==-1][['instance_id']]

#完整特征集
train_csr = sparse.hstack(
    (sparse.csr_matrix(data[data.click!=-1][row_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(data[data.click==-1][row_feature]), base_predict_csr), 'csr').astype('float32')


print(train_csr.shape)
del data
gc.collect()
xx_pre =[]
best_score = []

#利用lgb训练模型，8折
skf = StratifiedKFold(n_splits=8, random_state=3, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    print(index+1)
    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=200, reg_alpha=1, reg_lambda=1,# reg_alpha=0, reg_lambda=0.1,
        max_depth=-1, n_estimators=5000, objective='binary',
        subsample=0.7, colsample_bytree=0.5, subsample_freq=1,
        learning_rate=0.02, random_state=3# seed*10+index
    )
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=50,verbose = 50)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    xx_pre.append(test_pred)
print(np.mean(xx_pre))

#选择8折中验证分数最好的前8折生成最后预测结果
s=0
loss=0
n=8
for i in np.argsort(best_score)[:n]:
    s += xx_pre[i]
    loss += best_score[i]
s = s/n
loss = loss/n
predict_result = iid
predict_result['predicted_score'] = s
predict_result[['instance_id', 'predicted_score']].to_csv("./data/sub2.csv", index=False)
