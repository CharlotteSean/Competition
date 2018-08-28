import numpy as np
import pandas as pd
from datetime import datetime

import gc
gc.collect()
train = pd.read_csv('train1.csv',dtype={0:np.uint32, 1:np.uint8, 3:np.uint16, 5:np.uint16})
test = pd.read_csv('test.csv',dtype={0:np.uint32, 2:np.uint16, 4:np.uint16})

gc.collect()
def feature1(data, test=False):
    gc.collect()
    if test == 1:
        dealed_data = data.drop_duplicates(['file_id'])['file_id']
    else:
        dealed_data = data.drop_duplicates(['file_id'])[['file_id','label']]
    # 提取各类别特征的类别数
    dealed_train = train.groupby("file_id")[["api",'tid','return_value']].nunique()
    dealed_train.columns = ['api_nunique','tid_nunique','value_nunique']

    temp = train.groupby(['file_id'])['index'].count().rename('id_count')
    dealed_train = pd.concat([dealed_train, temp],axis=1)

    # 对每个file_id,的每个feat计数，然后求其统计特征
    cate_feat = ['api','tid','return_value']
    for feat in cate_feat:
        temp = train.groupby(['file_id',feat])[feat].count().groupby(['file_id']).agg(['min','max','mean','median','std',pd.Series.mad,
                                                                            pd.Series.skew,pd.Series.kurt]).add_prefix(feat+'_cnt_')
        dealed_train = pd.concat([dealed_train, temp],axis=1)

    # 提取交叉特征，每个file_id，每个不同api的tid种类数的统计特征
    temp = train.groupby(['file_id','api'])['tid'].nunique().groupby('file_id').agg(['min','max','mean','median','std',pd.Series.mad,
                                                                    pd.Series.skew,pd.Series.kurt]).add_prefix('api_tid_')
    dealed_train = pd.concat([dealed_train, temp],axis=1)

    # 提取交叉特征，每个file_id，每个不同api的return种类数的统计特征
    temp = train.groupby(['file_id','api'])['return_value'].nunique().groupby('file_id').agg(['min','max','mean','median','std',pd.Series.mad,
                                                                    pd.Series.skew,pd.Series.kurt]).add_prefix('api_value_')                                                               
    dealed_train = pd.concat([dealed_train, temp],axis=1)

    temp = train.groupby(['file_id','tid'])['api'].nunique().groupby('file_id').agg(['min','max','mean','median','std',pd.Series.mad,
                                                                    pd.Series.skew,pd.Series.kurt]).add_prefix('tid_tid_')
    dealed_train = pd.concat([dealed_train, temp],axis=1)

    temp = train.groupby(['file_id','tid'])['return_value'].nunique().groupby('file_id').agg(['min','max','mean','median','std',pd.Series.mad,
                                                                    pd.Series.skew,pd.Series.kurt]).add_prefix('tid_value_')                                                               
    dealed_train = pd.concat([dealed_train, temp],axis=1)
    gc.collect()
    return dealed_data

train_data = feature1(train, test=False)
test_data = feature1(test, test=True)
train_data.to_csv('train_feature1.csv', index=False)
test_data.to_csv('test_feature1.csv', index=False)