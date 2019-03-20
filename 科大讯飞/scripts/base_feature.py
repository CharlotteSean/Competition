# -*- coding: utf-8 -*-
"""
@author: taopeng
"""

from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import warnings
import time
import pandas as pd
import gc
import numpy as np
import time
import gc
import os
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from lightgbm import plot_importance
import sklearn
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
warnings.filterwarnings("ignore")


bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
ad_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_0', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']
media_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']
origin_feat = ad_feature + media_feature + content_feature
num_feature = ['creative_width', 'creative_height', 'hour','area']
items = origin_feat+num_feature

#数据预处理
#将bool型数据转化为数值形式,并将类别数据进行编码
def processing_data(data):
    for i in bool_feature:
        data[i] = data[i].astype(int)
    data['advert_industry_inner_0'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
    #对类别特征进行编码
    for i in origin_feat:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    data.drop(['os_name'],axis=1,inplace=True)
    #计算面积
    data['area'] = data['creative_height'] * data['creative_width']
    #提取时间特征
    data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
    data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
    data['click'] = data['click'].astype(int)
    return data

    
#使用lgb模型的特征重要性提取前n个特征
def feat_topn(data,n,label):
    train_x,val_x,train_y,val_y = train_test_split(data.drop([label],axis=1),data[label],test_size=0.3,random_state = 42)
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'reg_alpha':1,
        'reg_lambda':1,
       
        'subsample':0.7,
        'colsample_bytree':0.7,
        'learning_rate':0.05,
        'min_child_weight':1,
        'num_leaves':63,
       
        'verbose': 0,
        'silent':True
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=0)
    y_pred = gbm.predict(val_x, num_iteration=gbm.best_iteration)
    featscore= pd.DataFrame(
    {'feat': train_x.columns,
     'gain': list(gbm.feature_importance(importance_type='gain'))}).sort_values('gain',ascending=False)
 #   print(featscore[:n])
    feat = featscore.iloc[:n]['feat']
    return feat

#构造变量之间的交互特征，比如每个adid广告被投放在几个不同的城市；由于交互特征较多，进行筛选
def get_cross_feat(data,items):
    cross_feat = []
    n =3
    for col in tqdm(items):
        cols =[i for i in items if i != col]
        tmp = data.groupby(col)[cols].agg({'nunique'})
        tmp.columns = [col+'_'+'_'.join(i) for i in tmp.columns]
        tmp.drop(tmp.columns[tmp.nunique()==1],axis=1,inplace=True)
        
        cross_feat.extend(list(tmp.columns))
        tmp = tmp.reset_index()
        df = pd.merge(data[data['click']!=-1][[col,'click']],tmp,on=col,how='left')
        feat = list(feat_topn(df.drop(col,axis=1),n , 'click'))
        data = pd.merge(data,tmp[feat+[col]],on=col,how='left')
    return data,cross_feat


#对每个类做计数特征
def get_count_feat(data,items):
    count_feat = []
    for col in origin_feat:
        value_counts = data.groupby(col)[col].count().to_frame(col+'_count').reset_index()
        data = pd.merge(data,value_counts,on=col,how='left')
        count_feat.append(col+'_count')
    return data,count_feat


if __name__ == '__main__':
    train_chusai = pd.read_table('../data/round1_iflyad_train.txt')
    train = pd.read_table('./data/round2_iflyad_train.txt')
    test = pd.read_table( './data/round2_iflyad_test_feature.txt')
    data = pd.concat([train, test], axis=0, ignore_index=True)
    data = pd.concat([train_chusai, data], axis=0, ignore_index=True)
    print(len(data))   
    data = data.fillna(-1)
    data = processing_data(data)
    data, cross_feat = get_cross_feat(data,items)
    data, count_feat = get_count_feat(data,items)
    all_feat = num_feature +list(count_feat)+list(cross_feat)
    
    predict = data[data.click == -1]
    predict_result = predict[['instance_id']]
    predict_result['predicted_score'] = 0
    predict_x = predict.drop('click', axis=1)
    
    train_x = data[data.click != -1]
    train_y = data[data.click != -1].click.values
    
    base_train_csr = sparse.csr_matrix((len(train_x), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))
    print('one-hot prepared')
    #对类别特征进行onehot处理
    enc = OneHotEncoder()
    for feature in origin_feat:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr', 'bool')
    print('one-hot over !')
    #对用户标签进行cv
    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                    'bool')
    print('cv over !')
    #利用chi筛选特征
    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(base_train_csr, train_y)
    base_train_csr = feature_select.transform(base_train_csr)
    base_predict_csr = feature_select.transform(base_predict_csr)
    #组合所有基本特征
    base_train_csr = sparse.hstack(
            (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
    base_predict_csr = sparse.hstack(
            (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
    #保存基本特征 
    sparse.save_npz('./data/base_train_csr.npz', base_train_csr)
    sparse.save_npz('./data/base_predict_csr.npz', base_predict_csr)

    
    
        
    
    
    

    
    
