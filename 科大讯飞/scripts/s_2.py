# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:10:11 2018

@author: SY
"""

from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import warnings
import time
import pandas as pd
import gc
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

train_chusai = pd.read_table('../data/round1_iflyad_train.txt')
train = pd.read_table('./data/round2_iflyad_train.txt')
test = pd.read_table( './data/round2_iflyad_test_feature.txt')

data = pd.concat([train, test], axis=0, ignore_index=True)
data = pd.concat([train_chusai, data], axis=0, ignore_index=True)
print(len(data))

data = data.fillna(-1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)
del data['click']

bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
for i in bool_feature:
    data[i] = data[i].astype(int)
    
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']
origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

data.drop(['time','os_name'],axis=1,inplace=True)
cate_feature = origin_cate_list
num_feature = ['creative_width', 'creative_height', 'hour','area']

feature = cate_feature + num_feature
print(len(feature), feature)

data['area'] = data['creative_height'] * data['creative_width']


#计算交叉特征并进行cpa
#先pca，再merge
origin_feat = origin_cate_list+num_feature
cross_feat = []
pca = PCA(n_components=2)
for col in origin_feat:
    print(col)
    cols =[i for i in origin_feat if i != col]
    cross_nunique = data.groupby(col)[cols].agg({'nunique'})
    cross_nunique.columns = [col+'_'+'_'.join(i) for i in cross_nunique.columns]
    cross_pca =pca.fit_transform(cross_nunique)
    cross_pca = pd.DataFrame(cross_pca,columns=[col+'_pca_'+str(i) for i in [0,1]])
    cross_feat.extend(cross_pca.columns)
    cross_pca.insert(0,col,cross_nunique.index)
    del cross_nunique
    gc.collect()
    data = pd.merge(data,cross_pca,on=col,how='left')

count_feat = []
#计算每种类别的数量
for col in origin_feat:
        value_counts = data.groupby(col)[col].count().to_frame(col+'_count').reset_index()
        data = pd.merge(data,value_counts,on=col,how='left')
        count_feat.append(col+'_count')
        
num_feature = num_feature+count_feat+list(cross_feat)
data.shape


'''eda'''
data['osv'] = data['osv'].astype(str)
data['osv'] = data['osv'].apply(lambda x:x.replace('_', '.').replace('Android', '').replace('android', '').replace('iOS', '').replace('iPhone OS', '').replace('iPhoneOS', '').strip(' '))

data['model'] = data['model'].fillna('-1').astype(str)
data['model'] = data['model'].apply(lambda x:x.replace('+', ' ').replace('-', ' ').replace(',', ' ').replace('%', ' ').lower())





predict = data[data.label == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)

train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

base_train_csr = sparse.csr_matrix((len(train_x), 0))
base_predict_csr = sparse.csr_matrix((len(predict_x), 0))


enc = OneHotEncoder()
for feature in cate_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                   'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                     'csr',
                                     'bool')
print('one-hot prepared !')

cv = CountVectorizer(min_df=20)
for feature in ['user_tags']:
    data[feature] = data[feature].astype(str)
    cv.fit(data[feature])
    base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                     'bool')
print('cv prepared !')


feature_select = SelectPercentile(chi2, percentile=95)
feature_select.fit(base_train_csr, train_y)
base_train_csr = feature_select.transform(base_train_csr)
base_predict_csr = feature_select.transform(base_predict_csr)
base_train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
base_predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')

sparse.save_npz('./feature/base_train_csr.npz', base_train_csr)
sparse.save_npz('./feature/base_predict_csr.npz', base_predict_csr)
