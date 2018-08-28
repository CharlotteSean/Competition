import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
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
        polynm = PolynomialFeatures(degree=2, interaction_only=True)
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
def add_avg(df):
    array = np.array(df["平均功率"])
    newarray=[]
    num = 0
    for i in np.arange(len(array)):
        for j in np.arange(10):
            if i<10:
                num = (array[j-1]+array[j-2]+array[j-3])/3
            if i>=10:
                num = (array[i-1]+array[i-2]+array[i-3]+array[i-5]+array[i-6]+array[i-7]+array[i-8]+array[i-9])/9
        newarray.append(num)
    df["old平均功率"] = newarray
    return df

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
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
bad_feature = ['ID','功率A', '功率B', '功率C', '平均功率', '现场温度', '电压A', '电压B', '电压C', '电流B', '电流C', '转换效率', '转换效率A', '转换效率B', '转换效率C']
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
    
    replace_value = (all_data.loc[index - before_offset, col_index].values + all_data.loc[index + after_offset, col_index].values) /2
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


X_train, X_test, y_train, y_test, sub_data, sm, polynm = generate_train_data(train_data, test_data, poly=True, select=True)

clean_X_train, clean_X_test, clean_y_train, clean_y_test, clean_sub_data, _, _ = generate_train_data(cleaned_train_data, cleaned_sub_data, poly=False, select=False)

clean_X = np.concatenate([clean_X_train, clean_X_test])
clean_y = np.concatenate([clean_y_train, clean_y_test])
clean_X = polynm.transform(clean_X)
clean_X = sm.transform(clean_X)

clean_sub_data = polynm.transform(clean_sub_data)
clean_sub_data = sm.transform(clean_sub_data)

all_X_train = np.concatenate([X_train, X_test])
all_y_train = np.concatenate([y_train, y_test])

lgb1 = LGBMRegressor(n_estimators=300, num_leaves=31, random_state=5,learning_rate=0.05) 
lgb2 = LGBMRegressor(n_estimators=150, num_leaves=31, random_state=5,learning_rate=0.05) 
lgb3 = LGBMRegressor(n_estimators=300, num_leaves=15, random_state=9,learning_rate=0.05)

lgb4 = LGBMRegressor(n_estimators=900, max_depth=5, random_state=5, n_jobs=8) 
lgb5 = LGBMRegressor(n_estimators=850, max_depth=4, random_state=7, n_jobs=8)
lgb6 = LGBMRegressor(n_estimators=720, max_depth=4, random_state=9, n_jobs=8)

xgbt1 = xgb.XGBRegressor(n_estimators=950, max_depth=3, max_features='sqrt', random_state=321, n_jobs=8)
xgbt2 = xgb.XGBRegressor(n_estimators=1000, max_depth=3, max_features='sqrt', random_state=456, n_jobs=8)
xgbt3 = xgb.XGBRegressor(n_estimators=1100, max_depth=3, max_features='sqrt', random_state=789, n_jobs=8)

# n_estimators=1000  max_depth=5  'sqrt'  GradientBoostingRegressor 最佳参数 ,learning_rate=0.08
gbdt1 = GradientBoostingRegressor(n_estimators=800, max_depth=4, max_features='log2', random_state=123,learning_rate=0.08)
gbdt2 = GradientBoostingRegressor(n_estimators=900, max_depth=4, max_features='log2', random_state=456,learning_rate=0.08)
gbdt3 = GradientBoostingRegressor(n_estimators=1000, max_depth=5, max_features='log2', random_state=789,learning_rate=0.08)

# n_estimators=700, max_features='auto', random_state=2, n_jobs=8,max_depth=10

# xgbt1 = xgb.XGBRegressor(n_estimators=950, max_depth=3, max_features='sqrt', random_state=2, n_jobs=8)
# xgbt2 = xgb.XGBRegressor(n_estimators=1000, max_depth=3, max_features='sqrt', random_state=3, n_jobs=8)
# xgbt3 = xgb.XGBRegressor(n_estimators=1100, max_depth=3, max_features='sqrt', random_state=4, n_jobs=8)

# gbdt1 = GradientBoostingRegressor(n_estimators=500, max_depth=3, max_features='sqrt', random_state=2)
# gbdt2 = GradientBoostingRegressor(n_estimators=400, max_depth=3, max_features='sqrt', random_state=3)
# gbdt3 = GradientBoostingRegressor(n_estimators=500, max_depth=4, max_features='log2', random_state=4)

# forest1 = RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=2, n_jobs=8)
# forest2 = RandomForestRegressor(n_estimators=300, max_features='log2', random_state=3, n_jobs=8)
# forest3 = RandomForestRegressor(n_estimators=600, max_features='sqrt', random_state=4, n_jobs=8) 

# lgb1 = LGBMRegressor(n_estimators=900, max_depth=5, random_state=2, n_jobs=8) 
# lgb2 = LGBMRegressor(n_estimators=850, max_depth=4, random_state=3, n_jobs=8)
# lgb3 = LGBMRegressor(n_estimators=720, max_depth=4, random_state=4, n_jobs=8)

cross_validation_test(
    models=[    
        lgb4, lgb5, lgb6,
        xgbt1,xgbt2,xgbt3,
        gbdt1, gbdt2, gbdt3,
        lgb1, lgb2, lgb3
    ],
    train_X_data=[
        all_X_train, all_X_train, all_X_train, all_X_train,
        all_X_train, all_X_train, all_X_train, all_X_train,
        all_X_train, all_X_train, all_X_train, all_X_train
    ],
    train_y_data=[
        all_y_train, all_y_train, all_y_train, all_y_train,
        all_y_train, all_y_train, all_y_train, all_y_train,
        all_y_train, all_y_train, all_y_train, all_y_train
    ],cv=10
)
# cross_validation_test(
#     models=[    
#         xgbt1, xgbt2, xgbt3,
#         gbdt1, gbdt2, gbdt3,
#         forest1, forest2, forest3,
#         lgb1, lgb2, lgb3
#     ],
#     train_X_data=[
#         all_X_train, all_X_train, all_X_train, all_X_train,
#         all_X_train, all_X_train, all_X_train, all_X_train,
#         all_X_train, all_X_train, all_X_train, all_X_train
#     ],
#     train_y_data=[
#         all_y_train, all_y_train, all_y_train, all_y_train,
#         all_y_train, all_y_train, all_y_train, all_y_train,
#         all_y_train, all_y_train, all_y_train, all_y_train
#     ]
# )
"""
Model	Avg MSE	Avg Score
0	1.XGBRegressor	0.029151	0.868166
1	2.XGBRegressor	0.029116	0.868254
2	3.XGBRegressor	0.029026	0.868511
3	4.GBRegressor	0.029010	0.869202
4	5.GBRegressor	0.027673	0.871483
5	6.GBRegressor	0.026850	0.874885
6	7.RFRegressor	0.030454	0.865121
7	8.RFRegressor	0.030332	0.864875
8	9.RFRegressor	0.030499	0.864881
9	10.LGBMRegressor	0.029452	0.867472
10	11.LGBMRegressor	0.029263	0.868362
11	12.LGBMRegressor	0.029340	0.868164
"""

regrs = [
    xgbt1, gbdt1, lgb1,
    lgb4, lgb5, lgb6,
    xgbt2, gbdt2, lgb2,
    xgbt3, gbdt3, lgb3
]

class Stacker(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
    
    # X: 原始训练集, y: 原始训练集真实值, predict_data: 原始待预测数据
    def fit_predict(self, X, y, predict_data):
        X = np.array(X)
        y = np.array(y)
        T = np.array(predict_data)

        folds = list(KFold(n_splits=self.n_splits, shuffle=False, random_state=2018).split(X, y))
        
        # 以基学习器预测结果为特征的 stacker的训练数据 与 stacker预测数据
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_predict = np.zeros((T.shape[0], len(self.base_models)))
        
        for i, regr in enumerate(self.base_models):
            print(i + 1, 'Base model:', str(regr).split('(')[0])
            S_predict_i = np.zeros((T.shape[0], self.n_splits))
            
            for j, (train_idx, test_idx) in enumerate(folds):
                # 将X分为训练集与测试集
                X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                print ('Fit fold', (j+1), '...')
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)                
                S_train[test_idx, i] = y_pred
                S_predict_i[:, j] = regr.predict(T)
            
            S_predict[:, i] = S_predict_i.mean(axis=1)

        nmse_score = cross_val_score(self.stacker, S_train, y, cv=5, scoring='neg_mean_squared_error')
        print('CV MSE:', -nmse_score)
        print('Stacker AVG MSE:', -nmse_score.mean(), 'Stacker AVG Score:', np.mean(np.divide(1, 1 + np.sqrt(-nmse_score))))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_predict)
        return res, S_train, S_predict

# stacking_mode1 = Ridge(alpha=0.008, copy_X=True, fit_intercept=False, solver='auto', random_state=2)
stacking_model = SVR(C=100, gamma=0.01, epsilon=0.01)
stacker = Stacker(10, stacking_model, regrs)
pred_stack, S_train_data, S_predict_data = stacker.fit_predict(all_X_train, all_y_train, sub_data)

stacking_model2 = SVR(C=100, gamma=0.01, epsilon=0.01)
stacker2 = Stacker(10, stacking_model2, regrs)
pred_clean_stack, S_clean_train_data, S_clean_predict_data = stacker2.fit_predict(clean_X, clean_y, clean_sub_data)

df_result['score'] = pred_stack
index = df_result[df_result['ID'].isin(special_missing_ID)].index
df_result.loc[index, 'score'] = 0.379993053
c_index = df_result[df_result['ID'].isin(cleaned_sub_data_ID)].index
df_result.loc[c_index, 'score'] = pred_clean_stack
df_result.to_csv('sub_082002.csv', index=False, header=False)

