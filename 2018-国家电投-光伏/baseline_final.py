import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 

from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import Ridge,Lasso,RidgeCV,LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras import layers
# from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import optimizers

import xgboost as xgb
from lightgbm import LGBMRegressor
import math

# 数据预处理
def drop_all_outlier(df):
    df.drop_duplicates(df.columns.drop('ID'), keep='first', inplace=True)
    df.drop(df[(df.电压A > 800) | (df.电压A < 500)].index,inplace=True)
    df.drop(df[(df.电压B > 800) | (df.电压B < 500)].index,inplace=True)
    df.drop(df[(df.电压C > 800) | (df.电压C < 500)].index,inplace=True)
    df.drop(df[(df.现场温度 > 30) | (df.现场温度 < -30)].index,inplace=True)
    df.drop(df[(df.转换效率A > 100)].index,inplace=True)
    df.drop(df[(df.转换效率B > 100)].index,inplace=True)
    df.drop(df[(df.转换效率C > 100)].index,inplace=True)
    df.drop(df[(df.功率A > 6000)].index,inplace=True)
    df.drop(df[(df.功率B > 6000)].index,inplace=True)
    df.drop(df[(df.功率C > 6000)].index,inplace=True)
    df.drop(df[(df.平均功率 > 6000)].index,inplace=True)
    df.drop(df[(df.风向 > 360)].index,inplace=True)
    df.drop(df[(df.风速 > 20)].index,inplace=True)
    return df
# 生成数据
def generate_train_data(train_data, test_data, poly=False, select=False):
    y = train_data['发电量']
    X = train_data.drop(['发电量','ID'], axis=1)
    sub_data = test_data.drop(['ID'], axis=1)
    
    polynm = None
    if poly:
        from sklearn.preprocessing import PolynomialFeatures
        polynm = PolynomialFeatures(degree=2, interaction_only=False)
        X = polynm.fit_transform(X)
        sub_data = polynm.transform(sub_data)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    sm = None
    if select:
        from sklearn.feature_selection import SelectFromModel
        sm = SelectFromModel(GradientBoostingRegressor(random_state=2))
        X_train = sm.fit_transform(X_train, y_train)
        X_test = sm.transform(X_test)
        sub_data = sm.transform(sub_data)
        
    return X_train, X_test, y_train, y_test, sub_data, sm, polynm

def cal_score(mse):
    if isinstance(mse, float):
        return 1 / (1 + math.sqrt(mse))
    else:
        return np.divide(1, 1 + np.sqrt(mse))
#  定义交叉验证函数  
def cross_validation_test(models, train_X_data, train_y_data, cv=5):
    model_name, mse_avg, score_avg = [], [], []
    for i, model in enumerate(models):
        print(i + 1,'- Model:', str(model).split('(')[0])
        model_name.append(str(i + 1) + '.' + str(model).split('(')[0])
        nmse = cross_val_score(model, train_X_data[i], train_y_data[i], cv=cv, scoring='neg_mean_squared_error')
        
        avg_mse = np.average(-nmse)
        scores = cal_score(-nmse)
        avg_score = np.average(scores)
        mse_avg.append(avg_mse)
        score_avg.append(avg_score)
        print('MSE:', -nmse)
        print('Score:', scores)
        print('Average XGB - MSE:', avg_mse, ' - Score:', avg_score, '\n')
    res = pd.DataFrame()
    res['Model'] = model_name
    res['Avg MSE'] = mse_avg
    res['Avg Score'] = score_avg
    return res

# def add_newid(df):
#     ID = df["ID"]
#     df["new_id"]=(np.mod(ID,205))
#     return df
# def add_avg(df):
#     array = np.array(df["平均功率"])
#     newarray=[]
#     num = 0
#     for i in np.arange(len(array)):
#         for j in np.arange(10):
#             if i<10:
#                 num = (array[j-1]+array[j-2]+array[j-3])/3
#             if i>=10:
#                 num = (array[i-1]+array[i-2]+array[i-3]+array[i-5]+array[i-6]+array[i-7]+array[i-8]+array[i-9])/9
#         newarray.append(num)
#     df["old平均功率"] = newarray
#     return df
def add_avg(df):
    array = np.array(df["平均功率"])
    newarray=[]
    num = 0
    for i in np.arange(len(array)):
        if i<3:
            num = array[i]
        if i>=3:
            num = (array[i-1]+array[i-2]+array[i-3])/3
        if i>=9:
            num = (array[i-1]+array[i-2]+array[i-3]+array[i-5]+array[i-6]+array[i-7]+array[i-8]+array[i-9])/10
        newarray.append(num)
    df["old平均功率"] = newarray
    return df

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# feature = ['ID','电压A','电压B','电压C','电流A','电流B','电流C','功率A','功率B','功率C','平均功率']
# feature_train = feature+['发电量']
# train_data = train_data[feature_train]
# test_data = test_data[feature]
# train_data = add_newid(train_data)
# test_data = add_newid(test_data)
# train_data = add_avg(train_data)
# test_data = add_avg(test_data)
df_result = pd.DataFrame()
df_result['ID'] = list(test_data['ID'])
special_missing_ID = test_data[test_data[(test_data == 0) | (test_data == 0.)].count(axis=1) > 13]['ID']

cleaned_train_data = train_data.copy()
cleaned_train_data = drop_all_outlier(cleaned_train_data)

cleaned_sub_data = test_data.copy()
cleaned_sub_data = drop_all_outlier(cleaned_sub_data)
cleaned_sub_data_ID = cleaned_sub_data['ID']

# all_data = pd.concat([train_data, test_data], axis=0).sort_values(by='ID').reset_index().drop(['index'], axis=1)
# bad_feature = ['ID', '功率A', '功率B', '功率C', '平均功率', '现场温度', '电压A', '电压B', '电压C', '电流B', '电流C', '转换效率', '转换效率A', '转换效率B', '转换效率C']
# bad_index = all_data[bad_feature][
#     (all_data[bad_feature] > all_data[bad_feature].mean() + 2 * all_data[bad_feature].std()) | 
#     (all_data[bad_feature] < all_data[bad_feature].mean() - 2 * all_data[bad_feature].std())
# ].dropna(how='all').index

all_data  = pd.concat([train_data, test_data], axis=0).sort_values(by='ID').reset_index().drop(['index'], axis=1)
# bad_feature = ['ID','功率A', '功率B', '功率C', '平均功率', '现场温度', '电压A', '电压B', '电压C', '电流B', '电流C', '转换效率', '转换效率A', '转换效率B', '转换效率C']
bad_feature = ['ID','电压A','电压B','电压C','电流A','电流B','电流C','功率A','功率B','功率C','平均功率']

bad_index1 = all_data[bad_feature][
    (all_data[bad_feature] > all_data[bad_feature].mean() + 2 * all_data[bad_feature].std()) | 
    (all_data[bad_feature] < all_data[bad_feature].mean() - 2 * all_data[bad_feature].std())
].dropna(how='all').index

bad_index2 = all_data[
    ((all_data['电压A']<500)&(all_data['电压A']!=0))|
    ((all_data['电压B']<500)&(all_data['电压B']!=0))|
    ((all_data['电压C']<500)&(all_data['电压C']!=0))].index
bad_index = pd.Int64Index(list(bad_index1)+list(bad_index2))
# all_data.loc[np.concatenate([bad_index -1,bad_index,bad_index+1])].sort_values(by='ID', ascending=True)

nn_bad_data = all_data.loc[np.concatenate([bad_index - 1, bad_index, bad_index + 1])].sort_values(by='ID', ascending=True).drop_duplicates()
bad_data = all_data.loc[bad_index].sort_values(by='ID', ascending=True)

# 上下记录均值替代异常值
for idx, line in bad_data.iterrows():
    ID = line['ID']
    col_index = line[bad_feature][ 
        (line[bad_feature] > all_data[bad_feature].mean() + 3 * all_data[bad_feature].std())| 
        (line[bad_feature] < all_data[bad_feature].mean() - 3 * all_data[bad_feature].std())
    ].index
    index = all_data[all_data['ID'] == ID].index
    
    before_offset = 1
    while (idx + before_offset)in bad_index:
        before_offset += 1

    after_offset = 1
    while (idx + after_offset) in bad_index:
        after_offset += 1
    
    replace_value = (all_data.loc[index - before_offset, col_index].values + all_data.loc[index + after_offset, col_index].values) / 2
    all_data.loc[index, col_index] = replace_value[0]

#拆分数据
train_data = all_data.drop(all_data[all_data['ID'].isin(df_result['ID'])].index).reset_index().drop(['index'], axis=1)
test_data = all_data[all_data['ID'].isin(df_result['ID'])].drop(['发电量'], axis=1).reset_index().drop(['index'], axis=1)
len(train_data), len(test_data)
# 去除重复值
train_data = train_data.drop_duplicates(train_data.columns.drop('ID'), keep='first')

train_data = add_avg(train_data)
test_data = add_avg(test_data)
cleaned_train_data = add_avg(cleaned_train_data)
cleaned_sub_data = add_avg(cleaned_sub_data)

X_train, X_test, y_train, y_test, sub_data, sm, polynm = generate_train_data(train_data, test_data, poly=True, select=False)

clean_X_train, clean_X_test, clean_y_train, clean_y_test, clean_sub_data, _, _ = generate_train_data(cleaned_train_data, cleaned_sub_data, poly=False, select=False)

clean_X = np.concatenate([clean_X_train, clean_X_test])
clean_y = np.concatenate([clean_y_train, clean_y_test])
clean_X = polynm.transform(clean_X)
# clean_X = sm.transform(clean_X)

clean_sub_data = polynm.transform(clean_sub_data)
# clean_sub_data = sm.transform(clean_sub_data)

all_X_train = np.concatenate([X_train, X_test])
all_y_train = np.concatenate([y_train, y_test])

def cross_val(model,train_x, train_y,cv):
    nmse = cross_val_score(model, train_x,train_y, cv=cv, scoring='neg_mean_squared_error')
    avg_mse = np.average(-nmse)
    scores = cal_score(-nmse)
    avg_score = np.average(scores)

#     print('MSE:', -nmse)
#     print('Score:', scores)
    print(model)
    print('Average MSE:',avg_mse, ' - Score:', avg_score, '\n')

lgb1 = LGBMRegressor(n_estimators=300, num_leaves=31, random_state=5,learning_rate=0.05) 
lgb2 = LGBMRegressor(n_estimators=150, num_leaves=31, random_state=5,learning_rate=0.05) 
lgb3 = LGBMRegressor(n_estimators=300, num_leaves=15, random_state=9,learning_rate=0.05)

lgb4 = LGBMRegressor(n_estimators=900, max_depth=5, random_state=5,learning_rate=0.05) 
lgb5 = LGBMRegressor(n_estimators=850, max_depth=4, random_state=7,learning_rate=0.05)
lgb6 = LGBMRegressor(n_estimators=720, max_depth=4, random_state=9,learning_rate=0.05)

lgb7 = LGBMRegressor(n_estimators=150, num_leaves=31, random_state=5,learning_rate=0.05,colsample_bytree=1,subsample=0.7) 
lgb8 = LGBMRegressor(n_estimators=300, num_leaves=31, random_state=7,learning_rate=0.05,colsample_bytree=0.7,subsample=0.7)
lgb9 = LGBMRegressor(n_estimators=300, num_leaves=15, random_state=9,learning_rate=0.05,colsample_bytree=0.7,subsample=1)

lgb10 = LGBMRegressor(n_estimators=900, num_leaves=31, random_state=5,learning_rate=0.05,colsample_bytree=1,subsample=0.7) 
lgb11 = LGBMRegressor(n_estimators=850, num_leaves=31, random_state=7,learning_rate=0.05,colsample_bytree=0.7,subsample=0.7)
lgb12 = LGBMRegressor(n_estimators=720, num_leaves=15, random_state=9,learning_rate=0.05,colsample_bytree=0.7,subsample=1)
model_list = [lgb1,lgb2,lgb3,lgb4,lgb5,lgb6,lgb7,lgb8,lgb9,lgb10,lgb11,lgb12]
for mod in model_list:
    cross_val(mod, all_X_train, all_y_train,cv=10)

def lgb_cv(train_x, train_y, test_data):
    rmse_score = 0
    cvFold = 10
    result = df_result.copy()
    result['发电量']=0
    y_pred_test = 0
    for i in range(cvFold):
        print('第'+str(i)+'次交叉验证')
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=i)

        lgb_clf = LGBMRegressor(n_estimators=1500, num_leaves=31, random_state=7,
                                learning_rate=0.01,colsample_bytree=0.7,subsample=0.7)

        lgb_clf.fit(X_train, y_train, eval_metric='rmse', eval_set=(X_test, y_test), early_stopping_rounds=100)

        y_pred = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration_)
        score = mean_squared_error(y_test, y_pred)

        y_pred_test += lgb_clf.predict(test_data, num_iteration=lgb_clf.best_iteration_)
#         result['发电量'] += y_pred_test

        score += 1/(1 + math.sqrt(score))
        rmse_score += score

        print('测试集 RMSE：', score)

#     result['发电量'] = result['发电量']/cvFold
    y_pred_test /= cvFold
    print(rmse_score/cvFold)
    return y_pred_test

pred_stack1 = lgb_cv(all_X_train, all_y_train, sub_data)
pred_clean_stack1 = lgb_cv(clean_X, clean_y, clean_sub_data)

def xgb_cv(train_x, train_y, test_data):
    rmse_score = 0
    cvFold = 10
    result = df_result.copy()
    result['发电量']=0
    y_pred_test = 0
    for i in range(cvFold):
        print('第'+str(i)+'次交叉验证')
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=i)

        xgb_clf = xgb.XGBRegressor(n_estimators=1500, max_depth=4, random_state=7,n_jobs=8,
                                learning_rate=0.05,colsample_bytree=0.7,subsample=0.7)

        xgb_clf.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_test, y_test)], early_stopping_rounds=100)

        y_pred = xgb_clf.predict(X_test)
        score = mean_squared_error(y_test, y_pred)

        y_pred_test += xgb_clf.predict(test_data)
#         result['发电量'] += y_pred_test

        score += 1/(1 + math.sqrt(score))
        rmse_score += score

        print('测试集 RMSE：', score)

#     result['发电量'] = result['发电量']/cvFold
    y_pred_test /= cvFold
    print(rmse_score/cvFold)
    return y_pred_test

pred_stack6 = xgb_cv(all_X_train, all_y_train, sub_data)
pred_clean_stack6 = xgb_cv(clean_X, clean_y, clean_sub_data)

def gbdt_cv(train_x, train_y, test_data):
    rmse_score = 0
    cvFold = 10
    result = df_result.copy()
    result['发电量']=0
    y_pred_test = 0
    for i in range(cvFold):
        print('第'+str(i)+'次交叉验证')
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=i)
        gbdt = GradientBoostingRegressor(n_estimators=3000, max_depth=5, verbose=1,
                                          max_features='log2', random_state=123,learning_rate=0.01)
        gbdt.fit(X_train, y_train)

        y_pred = gbdt.predict(X_test)
        score = mean_squared_error(y_test, y_pred)

        y_pred_test += gbdt.predict(test_data)
#         result['发电量'] += y_pred_test
        score += 1/(1 + math.sqrt(score))
        rmse_score += score

        print('测试集 RMSE：', score)

#     result['发电量'] = result['发电量']/cvFold
    y_pred_test /= cvFold
    print(rmse_score/cvFold)
    return y_pred_test

pred_stack3 = gbdt_cv(all_X_train, all_y_train, sub_data)
pred_clean_stack3 = gbdt_cv(clean_X, clean_y, clean_sub_data)
# 0.9018241699817123

def pred_lgb_CV(all_train_x, all_train_y, test_data):
    rmse_score = 0
    cvFold = 5
    result = df_result.copy()
    result['发电量']=0
    y_pred_test = 0
    for num in range(4):
        kf = KFold(n_splits = 5, random_state=100*num + 10, shuffle=True)
        for train_ix, val_ix in kf.split(all_train_x, all_train_y):
            i = 0
            print('第',i,'次交叉')
            i += 1
            
            train_y = all_train_y[train_ix]
            train_x = all_train_x[train_ix]

            X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=num*20)

#             auc_score += list(lgb_clf.best_score_.values())[0]['auc']

            lgb_clf = LGBMRegressor(n_estimators=1500, num_leaves=31, random_state=7,
                                learning_rate=0.01,colsample_bytree=0.7,subsample=0.7)        
            lgb_clf.fit(X_train, y_train, eval_metric='rmse', eval_set=(X_test, y_test), early_stopping_rounds=100)
            y_pred = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration_)
            score = mean_squared_error(y_test, y_pred)

            y_pred_test += lgb_clf.predict(test_data, num_iteration=lgb_clf.best_iteration_)
#             result['发电量'] += y_pred_test
            score += 1/(1 + math.sqrt(score))
            rmse_score += score

            print('测试集 RMSE：', score)
#     result['发电量'] = result['发电量']/100
    y_pred_test /= cvFold
    print(rmse_score/cvFold)
    return y_pred_test

pred_stack2 = pred_lgb_CV(all_X_train, all_y_train, sub_data)
pred_clean_stack2 = pred_lgb_CV(clean_X, clean_y, clean_sub_data)

def all_cv(train_x, train_y, test_data):
    rmse_score = 0
    cvFold = 10
    result = df_result.copy()
    result['发电量']=0
    y_pred_test = 0
    y_pred_test1 = 0
    y_pred_test2 = 0
    y_pred_test3 = 0
    for i in range(cvFold):
        print('第'+str(i)+'次交叉验证')
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=i)
        # -------------------------------------------------------------------------
        gbdt = GradientBoostingRegressor(n_estimators=3000, max_depth=5, verbose=1,
                                          max_features='log2', random_state=123,learning_rate=0.01)
        gbdt.fit(X_train, y_train)

        y_pred1 = gbdt.predict(X_test)

        y_pred_test1 += gbdt.predict(test_data)
        # -------------------------------------------------------------------------
        
        lgb_clf = LGBMRegressor(n_estimators=1500, num_leaves=31, random_state=7,
                                learning_rate=0.01,colsample_bytree=0.7,subsample=0.7)

        lgb_clf.fit(X_train, y_train, eval_metric='rmse', eval_set=(X_test, y_test), early_stopping_rounds=100)

        y_pred2 = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration_)
        y_pred_test2 += lgb_clf.predict(test_data, num_iteration=lgb_clf.best_iteration_)
        # ------------------------------------------------------------------
        xgb_clf = xgb.XGBRegressor(n_estimators=1500, max_depth=4, random_state=7,n_jobs=8,
                                learning_rate=0.05,colsample_bytree=0.7,subsample=0.7)

        xgb_clf.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_test, y_test)], early_stopping_rounds=100)

        y_pred3 = xgb_clf.predict(X_test)

        y_pred_test3 += xgb_clf.predict(test_data)
        # -----------------------------------------------------------------
        score = mean_squared_error(y_test, (y_pred1+y_pred2+y_pred3)/3)

#         result['发电量'] += y_pred_test
        score += 1/(1 + math.sqrt(score))
        rmse_score += score

        print('测试集 RMSE：', score)

#     result['发电量'] = result['发电量']/cvFold
    y_pred_test = (y_pred_test1 + y_pred_test2)/2
    y_pred_test /= cvFold
    print(rmse_score/cvFold)
    return y_pred_test

pred_stack5 = all_cv(all_X_train, all_y_train, sub_data)
pred_clean_stack5 = all_cv(clean_X, clean_y, clean_sub_data)