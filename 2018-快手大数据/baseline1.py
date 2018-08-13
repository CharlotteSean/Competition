# --------------------------读取数据------------------------------------------
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

gc.collect()
begin = datetime.now()

# def data_order():
"""
    实现数据集变换....原始数据集header都没有...并且按id排序方便自己査看
"""
path = '/mnt/datasets/fusai/'
new_path = '/home/kesci/input/'

print('app_launch_log')
launch = pd.read_csv('app_launch_log.txt', sep='\t',
                  names=['user_id', 'launch_day'],
                  dtype={0: np.uint32, 1: np.uint8})
launch = launch.sort_values(by=['user_id', 'launch_day'])                

print('user_activity_log')
activity = pd.read_csv('user_activity_log.txt', sep='\t',
                    names=['user_id','activity_day','page','video_id','author_id','action_type'],
                    dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint32, 4: np.uint32, 5: np.uint8})
activity = activity.sort_values(by=['user_id','activity_day'])

print('user_register_log')
register = pd.read_csv('user_register_log.txt', sep='\t',
                    names=['user_id','register_day','register_type','device_type'],
                    dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint16})
register = register.sort_values(by=['user_id','register_day']) 

print('video_create_log')
video = pd.read_csv('video_create_log.txt', sep='\t',
                  names=['user_id', 'create_day'],
                  dtype={0: np.uint32, 1: np.uint8})
video = video.sort_values(by=['user_id', 'create_day']) 
print('read data done....',datetime.now() - begin)

# --------------------------------划分数据集--------------------------------
gc.collect()
def cut_data(begin_day, end_day):
    temp_launch = launch[(launch['launch_day'] >= begin_day) & (launch['launch_day'] <= end_day)]
    temp_activity = activity[(activity['activity_day'] >= begin_day) & (activity['activity_day'] <= end_day)]
    temp_register = register[(register['register_day'] >= begin_day) & (register['register_day'] <= end_day)]
    temp_create = video[(video['create_day'] >= begin_day) & (video['create_day'] <= end_day)]

    return temp_launch,temp_activity,temp_register,temp_create
   
begin = datetime.now()   
print('开始划分数据集...')
# 以1-10数据預测11-17某用户是否活跃
begin_day = 1
end_day = 10
train1_launch,train1_activity,train1_register,train1_create = cut_data(begin_day, end_day)

begin_day = 11
end_day = 17
test1_launch,test1_activity,test1_register,test1_create = cut_data(begin_day, end_day)
print('第一部分数据集划分完成！')

# 以1-16数据預测17-23某用户是否活跃
begin_day = 1
end_day = 17
train1_launch,train1_activity,train1_register,train1_create = cut_data(begin_day, end_day)

begin_day = 18
end_day = 24
test1_launch,test1_activity,test1_register,test1_create = cut_data(begin_day, end_day)
print('第一部分数据集划分完成！')

# 以8-23数据預测24-30某用户是否活跃
begin_day = 1
end_day = 23
train2_launch,train2_activity,train2_register,train2_create = cut_data(begin_day, end_day)

begin_day = 24
end_day = 30
test2_launch,test2_activity,test2_register,test2_create = cut_data(begin_day, end_day)
print('第二部分数据集划分完成！')

# 测试集
# 以15-30数据预测31-37某用户是否活跃
# begin_day = 1
# end_day = 30
# train3_launch,train3_activity,train3_register,train3_create = cut_data(begin_day, end_day)
# print('第三部分数据集划分完成！', datetime.now() - begin)

# 做训练数据，提取用户ID，以及生成label
gc.collect()
begin = datetime.now()

# def test_id():
#     test_reg = register['user_id']
#     test_cre = video['user_id']
#     test_lau = launch['user_id']
#     test_act = activity['user_id']
#     test_id = pd.DataFrame()
#     test_id['user_id'] = np.unique(pd.concat([test_reg, test_cre, test_lau, test_act]))

#     return test_id

def get_train_label1():
#     train_reg = train1_register['user_id']
#     train_cre = train1_create['user_id']
#     train_lau = train1_launch['user_id']
#     train_act = train1_activity['user_id']
    train_data_id = np.unique(pd.concat([train1_create['user_id'], train1_launch['user_id'], train1_activity['user_id']]))

#     test_reg = test1_register['user_id']
#     test_cre = test1_create['user_id']
#     test_lau = test1_launch['user_id']
#     test_act = test1_activity['user_id']
    test_data_id = np.unique(pd.concat([test1_create['user_id'], test1_launch['user_id'], test1_activity['user_id']]))

    train_label = []
    for i in train_data_id:
        if i in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] = train_label
    return train_data

def get_train_label2():
#     train_reg = train2_register['user_id']
#     train_cre = train2_create['user_id']
#     train_lau = train2_launch['user_id']
#     train_act = train2_activity['user_id']
    train_data_id = np.unique(pd.concat([train2_create['user_id'], train2_launch['user_id'], train2_activity['user_id']]))

#     test_reg = test2_register['user_id']
#     test_cre = test2_create['user_id']
#     test_lau = test2_launch['user_id']
#     test_act = test2_activity['user_id']
    test_data_id = np.unique(pd.concat([test2_create['user_id'], test2_launch['user_id'], test2_activity['user_id']]))

    train_label = []
    for i in train_data_id:
        if i in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] = train_label
    return train_data

# 训练集1,1-16天的用户
print('生成训练集1的ID，和label')

train1_id = get_train_label1()

# 训练集2,8-23天的用户ID和label
print('生成训练集2的ID，和label')

train2_id = get_train_label2()

# 测试集的用户ID
test_id = pd.DataFrame()
test_id['user_id'] = np.unique(pd.concat([video['user_id'],launch['user_id'],activity['user_id']]))

print('用户ID和label生成：', datetime.now() - begin)

gc.collect()
begin = datetime.now()
video_all = video.copy()
register_all = register.copy()
launch_all = launch.copy()
activity_all = activity.copy()

# -----------------------------特征工程------------------------------------------------
def feature_extraction(data_id, begin_day, end_day):
    gc.collect()
    # --------------------------处理video特征----------------------------------
    begin = datetime.now()
    # 拍摄视频的次数
    video = video_all[(video_all['create_day'] >= begin_day) & (video_all['create_day'] <= end_day)]
    video_count = video.groupby(['user_id'])['create_day'].count().rename('create_num').reset_index()
    data_id = pd.merge(data_id, video_count, on='user_id', how='left')
    # 拍摄视频的天数
    video_days = video.drop_duplicates(['user_id','create_day']).groupby(['user_id']).size().rename('create_days').reset_index()
    data_id = pd.merge(data_id, video_days, on='user_id', how='left')

    # 拍摄视频的天数占比
    data_id['create_days_ratio'] = (data_id['create_days']/(end_day - begin_day + 1))*100

    # 平均每天拍摄视频的次数（/16）
    data_id['create_nums_ratio16'] = data_id['create_num']/(end_day - begin_day + 1)
    # 平均每拍摄视频的次数（/拍摄视频天数）
    data_id['create_nums_ratio'] = data_id['create_num']/data_id['create_days']
    # 拍摄视频日期的标准差,均值，中位数，偏度,众数
    video_std = video.groupby(['user_id'])['create_day'].std().rename('create_std').reset_index()
    data_id = pd.merge(data_id, video_std, on='user_id', how='left')
    video_mean = video.groupby(['user_id'])['create_day'].mean().rename('create_mean').reset_index()
    data_id = pd.merge(data_id, video_mean, on='user_id', how='left')
    video_median = video.groupby(['user_id'])['create_day'].median().rename('create_median').reset_index()
    data_id = pd.merge(data_id, video_median, on='user_id', how='left')
    video_skew = video.groupby(['user_id'])['create_day'].skew().rename('create_skew').reset_index()
    data_id = pd.merge(data_id, video_skew, on='user_id', how='left')
    video_mode = video.groupby(['user_id'])['create_day'].apply(mode).rename('create_mode').reset_index()
    video_mode['create_mode'] = video_mode['create_mode'].apply(lambda x:int(x[0]))
    data_id = pd.merge(data_id, video_mode, on='user_id', how='left')

    # 拍摄视频日期的最后一天（max），最早一天（min）
    video_max = video.groupby(['user_id'])['create_day'].max().rename('create_max').reset_index()
    data_id = pd.merge(data_id, video_max, on='user_id', how='left')
    video_min = video.groupby(['user_id'])['create_day'].min().rename('create_min').reset_index()
    data_id = pd.merge(data_id, video_min, on='user_id', how='left')
    # 最后一天，最早一天时间差
    data_id['create_span'] = data_id['create_max'] - data_id['create_min']
    # 创建视频最后一天距离预测日期的天数
    data_id['create_pre_cut_last_day'] = end_day + 1 - data_id['create_max']
    
    # 每天创建视频的数量
    create_num_day = video.groupby(['user_id','create_day']).size().rename('create_num_day').reset_index()

    # 一天中创建视频的最多，最少数量
    create_num_max_one_day =create_num_day.groupby(['user_id'])['create_num_day'].max().rename('create_num_max_one_day').reset_index()
    data_id = pd.merge(data_id, create_num_max_one_day, on='user_id', how='left')
    create_num_min_one_day = create_num_day.groupby(['user_id'])['create_num_day'].min().rename('create_num_min_one_day').reset_index()
    data_id = pd.merge(data_id, create_num_min_one_day, on='user_id', how='left')
    print('create_video特征提取完毕：',(datetime.now() - begin))
    
    # 统计最后7天的数据规律，和上述特征类似---------------------------------------------
    video7 = video[video['create_day'] >= (end_day - 6)]
    video7_count = video7.groupby(['user_id'])['create_day'].count().rename('create7_num').reset_index()
    data_id = pd.merge(data_id, video7_count, on='user_id', how='left')
    video7_days = video7.drop_duplicates(['user_id','create_day']).groupby(['user_id']).size().rename('create7_days').reset_index()
    data_id = pd.merge(data_id, video7_days, on='user_id', how='left')

    #-------------------------------------------------------------------------------
    
    #----------------------处理注册特征---------------------------
    # 设备类型大于400的，直接作为400处理，，因为后面的device太少了，
    begin = datetime.now()
    register = register_all[(register_all['register_day'] >= begin_day) & (register_all['register_day'] <= end_day)]
    register['device_type'][register['device_type'] >= 400] = 400
    # 注册日期距离预测日期的天数
    register['pre_cut_register_day'] = end_day + 1 - register['register_day']
    data_id = pd.merge(data_id, register, on='user_id', how='left')
    print('register特征提取完毕：',datetime.now() - begin)
    # -----------------------------------------------------------
    
    # -------------------处理登录特征-----------------------------
    # 登陆次数
    launch = launch_all[(launch_all['launch_day'] >= begin_day) & (launch_all['launch_day'] <= end_day)]
    launch_num = launch.groupby(['user_id'])['launch_day'].count().rename('launch_num').reset_index()
    data_id = pd.merge(data_id, launch_num, on='user_id', how='left')
    # 登录天数
    launch_days = launch.drop_duplicates(['user_id','launch_day']).groupby(['user_id']).size().rename('launch_days').reset_index()
    data_id = pd.merge(data_id, launch_days, on='user_id', how='left')
    # 登录天数占比(/16)
    data_id['launch_days_ratio16'] = (data_id['launch_days']/(end_day - begin_day + 1))*100
    # 平均每天登录的次数（/16）
    data_id['launch_nums_ratio16'] = data_id['launch_num']/(end_day - begin_day + 1)
    # 平均每天登录的次数（/拍摄视频天数）
    data_id['create_nums_ratio'] = data_id['launch_num']/data_id['launch_days']
    
    # 登录日期的标准差,均值，中位数，偏度,众数
    launch_mean = launch.groupby(['user_id'])['launch_day'].mean().rename('launch_mean').reset_index()
    data_id = pd.merge(data_id, launch_mean, on='user_id', how='left')
    launch_std = launch.groupby(['user_id'])['launch_day'].std().rename('launch_std').reset_index()
    data_id = pd.merge(data_id, launch_std, on='user_id', how='left')
    launch_var = launch.groupby(['user_id'])['launch_day'].var().rename('launch_var').reset_index()
    data_id = pd.merge(data_id, launch_var, on='user_id', how='left') 
    launch_median = launch.groupby(['user_id'])['launch_day'].median().rename('launch_median').reset_index()
    data_id = pd.merge(data_id, launch_median, on='user_id', how='left') 
    launch_skew = launch.groupby(['user_id'])['launch_day'].skew().rename('launch_skew').reset_index()
    data_id = pd.merge(data_id, launch_skew, on='user_id', how='left')
    launch_mode = launch.groupby(['user_id'])['launch_day'].apply(mode).rename('launch_mode').reset_index()
    launch_mode['launch_mode'] = launch_mode['launch_mode'].apply(lambda x:int(x[0]))
    data_id = pd.merge(data_id, launch_mode, on='user_id', how='left')
    
    # 登录日期的最后一天（max），最早一天（min）
    launch_max = launch.groupby(['user_id'])['launch_day'].max().rename('launch_max').reset_index()
    data_id = pd.merge(data_id, launch_max, on='user_id', how='left')
    launch_min = launch.groupby(['user_id'])['launch_day'].min().rename('launch_min').reset_index()
    data_id = pd.merge(data_id, launch_min, on='user_id', how='left')
    
    # 最后一天，最早一天时间差
    data_id['launch_span'] = data_id['launch_max'] - data_id['launch_min']
    
    # 登录最后一天距离预测日期的天数
    data_id['launch_pre_cut_last_day'] = end_day + 1 - data_id['launch_max']
    # 每天登陆的次数
    launch_num_day = launch.groupby(['user_id','launch_day']).size().rename('launch_num_day').reset_index()

    # 一天中登录的最多，最少次数
    launch_num_max_one_day = launch_num_day.groupby(['user_id'])['launch_num_day'].max().rename('launch_num_max_one_day').reset_index()
    data_id = pd.merge(data_id, launch_num_max_one_day, on='user_id', how='left')
    launch_num_min_one_day = launch_num_day.groupby(['user_id'])['launch_num_day'].min().rename('launch_num_min_one_day').reset_index()
    data_id = pd.merge(data_id, launch_num_min_one_day, on='user_id', how='left')
    print('launch特征提取完毕：', datetime.now() - begin)
    # --------------------------------------------------------
    
    # -----------------处理activity特征------------------------
    begin = datetime.now()
    # 行为总数
    activity = activity_all[(activity_all['activity_day'] >= begin_day) & (activity_all['activity_day'] <= end_day)]
    activity_num = activity.groupby('user_id')['activity_day'].count().rename('activity_num').reset_index()
    data_id = pd.merge(data_id, activity_num, on='user_id', how='left')
    # 有activity的总天数
    activity_days = activity.drop_duplicates(['user_id','activity_days']).groupby(['user_id'])['activity_day'].count().rename('activity_days').reset_index()
    data_id = pd.merge(data_id, activity_days, on='user_id', how='left')

    # 有行为的天数占比(/16)
    data_id['activity_days_ratio'] = (data_id['activity_days']/(end_day - begin_day + 1))*100

    # 平均每天行为的次数（/16）
    data_id['activity_nums_ratio16'] = data_id['activity_num']/(end_day - begin_day + 1)
    # 平均每行为的次数（/行为天数）
    data_id['activity_nums_ratio'] = data_id['activity_num']/data_id['activity_days']

    # 行为日期的var,mean,mode等特征（去重还是不去重呢）
    activity_day_std = activity.groupby(['user_id'])['activity_day'].std().rename('activity_day_std').reset_index()
    data_id = pd.merge(data_id, activity_day_std, on='user_id', how='left')
    activity_day_mean = activity.groupby(['user_id'])['activity_day'].mean().rename('activity_day_mean').reset_index()
    data_id = pd.merge(data_id, activity_day_mean, on='user_id', how='left')
    activity_day_median = activity.groupby(['user_id'])['activity_day'].median().rename('activity_day_median').reset_index()
    data_id = pd.merge(data_id, activity_day_median, on='user_id', how='left')
    activity_day_skew = activity.groupby(['user_id'])['activity_day'].skew().rename('activity_day_skew').reset_index()
    data_id = pd.merge(data_id, activity_day_skew, on='user_id', how='left')

    activity_day_mode = activity.groupby(['user_id'])['activity_day'].apply(mode).rename('activity_day_mode').reset_index()
    activity_day_mode['activity_day_mode'] = activity_day_mode['activity_day_mode'].apply(lambda x:int(x[0]))
    data_id = pd.merge(data_id, activity_day_mode, on='user_id', how='left')

    # 行为日期的最后一天（max）,最早一天（min）
    activity_day_max = activity.groupby(['user_id'])['activity_day'].max().rename('activity_day_max').reset_index()
    data_id = pd.merge(data_id, activity_day_max, on='user_id', how='left')
    activity_day_min = activity.groupby(['user_id'])['activity_day'].min().rename('activity_day_min').reset_index()
    data_id = pd.merge(data_id, activity_day_min, on='user_id', how='left')
    # 行为的最后一天和最早一天的时间差
    data_id['activity_day_span'] = data_id['activity_day_max'] - data_id['activity_day_min']
    # 行为的最后的一天距预测日期的天数
    data_id['activity_pre_cut_last_day'] = end_day + 1 - data_id['activity_day_max']

    # 每个用户，每天的行为次数
    activity_day_nums = activity.groupby(['user_id','activity_day'])['page'].count().rename('activity_day_nums').reset_index()
    # 求每天行为次数的mean,std,等
    activity_day_nums_max = activity_day_nums.groupby(['user_id'])['activity_day_nums'].max().rename('activity_day_nums_max').reset_index()
    data_id = pd.merge(data_id, activity_day_nums_max, on='user_id', how='left')
    activity_day_nums_min = activity_day_nums.groupby(['user_id'])['activity_day_nums'].min().rename('activity_day_nums_min').reset_index()
    data_id = pd.merge(data_id, activity_day_nums_min, on='user_id', how='left')
    activity_day_nums_mean = activity_day_nums.groupby(['user_id'])['activity_day_nums'].mean().rename('activity_day_nums_mean').reset_index()
    data_id = pd.merge(data_id, activity_day_nums_mean, on='user_id', how='left')
    activity_day_nums_var = activity_day_nums.groupby(['user_id'])['activity_day_nums'].var().rename('activity_day_nums_var').reset_index()
    data_id = pd.merge(data_id, activity_day_nums_var, on='user_id', how='left')
    activity_day_nums_mean = activity_day_nums.groupby(['user_id'])['activity_day_nums'].mean().rename('activity_day_nums_mean').reset_index()
    data_id = pd.merge(data_id, activity_day_nums_mean, on='user_id', how='left')

    # 每个用户每个video的观看次数
    activity_video_nums = activity.groupby(['user_id','video_id'])['page'].count().rename('activity_video_nums').reset_index()
    # 观看同一video的最大最小次数
    activity_video_nums_max = activity_video_nums.groupby(['user_id'])['activity_video_nums'].max().rename('activity_video_nums_max').reset_index()
    data_id = pd.merge(data_id, activity_video_nums_max, on='user_id', how='left')
    activity_video_nums_min = activity_video_nums.groupby(['user_id'])['activity_video_nums'].min().rename('activity_video_nums_min').reset_index()
    data_id = pd.merge(data_id, activity_video_nums_min, on='user_id', how='left')

    # 每个用户每个author的观看次数
    activity_author_nums = activity.groupby(['user_id','author_id'])['page'].count().rename('activity_author_nums').reset_index()
    # 观看同一author的最大最小次数
    activity_author_nums_max = activity_author_nums.groupby(['user_id'])['activity_author_nums'].max().rename('activity_video_nums_max').reset_index()
    data_id = pd.merge(data_id, activity_author_nums_max, on='user_id', how='left')
    activity_author_nums_min = activity_author_nums.groupby(['user_id'])['activity_author_nums'].min().rename('activity_video_nums_min').reset_index()
    data_id = pd.merge(data_id, activity_author_nums_min, on='user_id', how='left')

    # page0/1/2/3/4 次数统计
    page0_num = activity[activity['page'] == 0].groupby(['user_id'])['page'].count().rename('page0_num').reset_index()
    data_id = pd.merge(data_id, page0_num, on='user_id', how='left')
    page1_num = activity[activity['page'] == 1].groupby(['user_id'])['page'].count().rename('page1_num').reset_index()
    data_id = pd.merge(data_id, page1_num, on='user_id', how='left')
    page2_num = activity[activity['page'] == 2].groupby(['user_id'])['page'].count().rename('page2_num').reset_index()
    data_id = pd.merge(data_id, page2_num, on='user_id', how='left')
    page3_num = activity[activity['page'] == 3].groupby(['user_id'])['page'].count().rename('page3_num').reset_index()
    data_id = pd.merge(data_id, page3_num, on='user_id', how='left')
    page4_num = activity[activity['page'] == 4].groupby(['user_id'])['page'].count().rename('page4_num').reset_index()
    data_id = pd.merge(data_id, page4_num, on='user_id', how='left')
    
    # page0/1/2/3/4 次数,占总活动次数的百分比
    data_id['page0_percent'] = (data_id['page0_num']/data_id['activity_num'])*100
    data_id['page1_percent'] = (data_id['page1_num']/data_id['activity_num'])*100
    data_id['page2_percent'] = (data_id['page2_num']/data_id['activity_num'])*100
    data_id['page3_percent'] = (data_id['page3_num']/data_id['activity_num'])*100
    data_id['page4_percent'] = (data_id['page4_num']/data_id['activity_num'])*100

    # page0/1/2/3/4 页面，发生行为的最后一天
    page0_day_max = activity[activity['page'] == 0].groupby(['user_id'])['activity_day'].max().rename('page0_day_max').reset_index()
    data_id = pd.merge(data_id, page0_day_max, on='user_id', how='left')
    page1_day_max = activity[activity['page'] == 1].groupby(['user_id'])['activity_day'].max().rename('page1_day_max').reset_index()
    data_id = pd.merge(data_id, page1_day_max, on='user_id', how='left')
    page2_day_max = activity[activity['page'] == 2].groupby(['user_id'])['activity_day'].max().rename('page2_day_max').reset_index()
    data_id = pd.merge(data_id, page2_day_max, on='user_id', how='left')
    page3_day_max = activity[activity['page'] == 3].groupby(['user_id'])['activity_day'].max().rename('page3_day_max').reset_index()
    data_id = pd.merge(data_id, page3_day_max, on='user_id', how='left')
    page4_day_max = activity[activity['page'] == 4].groupby(['user_id'])['activity_day'].max().rename('page4_day_max').reset_index()
    data_id = pd.merge(data_id, page4_day_max, on='user_id', how='left')

    # 发生行为的最后一天距离预测日期的天数
    data_id['page0_pre_last'] = end_day + 1 - data_id['page0_day_max']
    data_id['page1_pre_last'] = end_day + 1 - data_id['page1_day_max']
    data_id['page2_pre_last'] = end_day + 1 - data_id['page2_day_max']
    data_id['page3_pre_last'] = end_day + 1 - data_id['page3_day_max']
    data_id['page4_pre_last'] = end_day + 1 - data_id['page4_day_max']

    # action_type0/1/2/3/4/5 行为次数统计
    action_type0_num = activity[activity['action_type'] == 0].groupby(['user_id'])['action_type'].count().rename('action_type0_num').reset_index()
    data_id = pd.merge(data_id, action_type0_num, on='user_id', how='left')
    action_type1_num = activity[activity['action_type'] == 1].groupby(['user_id'])['action_type'].count().rename('action_type1_num').reset_index()
    data_id = pd.merge(data_id, action_type1_num, on='user_id', how='left')
    action_type2_num = activity[activity['action_type'] == 2].groupby(['user_id'])['action_type'].count().rename('action_type2_num').reset_index()
    data_id = pd.merge(data_id, action_type2_num, on='user_id', how='left')
    action_type3_num = activity[activity['action_type'] == 3].groupby(['user_id'])['action_type'].count().rename('action_type3_num').reset_index()
    data_id = pd.merge(data_id, action_type3_num, on='user_id', how='left')
    action_type4_num = activity[activity['action_type'] == 4].groupby(['user_id'])['action_type'].count().rename('action_type4_num').reset_index()
    data_id = pd.merge(data_id, action_type4_num, on='user_id', how='left')
    action_type5_num = activity[activity['action_type'] == 5].groupby(['user_id'])['action_type'].count().rename('action_type5_num').reset_index()
    data_id = pd.merge(data_id, action_type5_num, on='user_id', how='left')

    # action_type0/1/2/3/4/5 行为次数占总次数的百分比
    data_id['action_type0_percent'] = (data_id['action_type0_num']/data_id['activity_num'])*100
    data_id['action_type1_percent'] = (data_id['action_type1_num']/data_id['activity_num'])*100
    data_id['action_type2_percent'] = (data_id['action_type2_num']/data_id['activity_num'])*100
    data_id['action_type3_percent'] = (data_id['action_type3_num']/data_id['activity_num'])*100
    data_id['action_type4_percent'] = (data_id['action_type4_num']/data_id['activity_num'])*100
    data_id['action_type5_percent'] = (data_id['action_type5_num']/data_id['activity_num'])*100

    # action_type0/1/2/3/4 页面，发生行为的最后一天
    action_type0_day_max = activity[activity['action_type'] == 0].groupby(['user_id'])['activity_day'].max().rename('action_type0_day_max').reset_index()
    data_id = pd.merge(data_id, action_type0_day_max, on='user_id', how='left')
    action_type1_day_max = activity[activity['action_type'] == 1].groupby(['user_id'])['activity_day'].max().rename('action_type1_day_max').reset_index()
    data_id = pd.merge(data_id, action_type1_day_max, on='user_id', how='left')
    action_type2_day_max = activity[activity['action_type'] == 2].groupby(['user_id'])['activity_day'].max().rename('action_type2_day_max').reset_index()
    data_id = pd.merge(data_id, action_type2_day_max, on='user_id', how='left')
    action_type3_day_max = activity[activity['action_type'] == 3].groupby(['user_id'])['activity_day'].max().rename('action_type3_day_max').reset_index()
    data_id = pd.merge(data_id, action_type3_day_max, on='user_id', how='left')
    action_type4_day_max = activity[activity['action_type'] == 4].groupby(['user_id'])['activity_day'].max().rename('action_type4_day_max').reset_index()
    data_id = pd.merge(data_id, action_type4_day_max, on='user_id', how='left')
    action_type5_day_max = activity[activity['action_type'] == 5].groupby(['user_id'])['activity_day'].max().rename('action_type5_day_max').reset_index()
    data_id = pd.merge(data_id, action_type5_day_max, on='user_id', how='left')
    
    # 发生行为的最后一天距离预测日期的天数
    data_id['action_type0_pre_last'] = end_day + 1 - data_id['action_type0_day_max']
    data_id['action_type1_pre_last'] = end_day + 1 - data_id['action_type1_day_max']
    data_id['action_type2_pre_last'] = end_day + 1 - data_id['action_type2_day_max']
    data_id['action_type3_pre_last'] = end_day + 1 - data_id['action_type3_day_max']
    data_id['action_type4_pre_last'] = end_day + 1 - data_id['action_type4_day_max']
    data_id['action_type5_pre_last'] = end_day + 1 - data_id['action_type5_day_max']

    data_id['action_type345'] = data_id['action_type3_num'] + data_id['action_type4_num'] + data_id['action_type5_num']
    
    print('activity特征提取完毕：',datetime.now() - begin)
    # --------------------------------------------------------
    return data_id

train1_feature = feature_extraction(train1_id, 1, 16)
train2_feature = feature_extraction(train2_id, 1, 23)
train = pd.concat([train1_feature, train2_feature],ignore_index=True)
train_y = train['label']
train_x = train.drop(['label','user_id'], axis=1)
test_data_id = pd.DataFrame()
test_data_id['user_id'] = test_id
test = feature_extraction(test_data_id, 1, 30)
test_x = test.drop(['user_id'], axis=1)

def predcit_cv():
    auc_score = 0
    for i in range(5):
        print('第'+str(i)+'次交叉验证')

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=i)

        lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=63, max_depth=-1,
                            learning_rate=0.01, n_estimators=700, max_bin=150,
                            min_child_weight=0.01, min_child_samples=20, subsample=0.7, subsample_freq=1,
                            colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=0.1, random_state=100*i+500, n_jobs=-1,
                            )

        lgb_clf.fit(X_train, y_train, eval_metric='auc', eval_set=(X_test, y_test), early_stopping_rounds=200)

#         print('训练集合F1：',train_score)

#         y_prob = lgb_clf.predict_proba(test_x)
    
#         test_data_id['prob'] = y_prob[:,1]
#         print('输出成功')
#         result.to_csv('submission_linxi_0724'+str(i)+'.csv', sep='\t',index=False, header=False)
        
        y_prob_test = lgb_clf.predict_proba(X_test,num_iteration=lgb_clf.best_iteration_)
        score = roc_auc_score(y_test, y_prob_test[:,1])
        auc_score += score

        print('测试集 AUC：',score)

    print(auc_score/5)

def predcit_result():
    lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=63, max_depth=-1,
                            learning_rate=0.01, n_estimators=800, max_bin=150,
                            min_child_weight=0.01, min_child_samples=20, subsample=0.9, subsample_freq=1,
                            colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=0.01, random_state=700, n_jobs=-1,
                            )
    print('开始训练')
    lgb_clf.fit(train_x, train_y, eval_metric='auc')
    
    plt.figure(figsize=(300,200))
    lgb.plot_importance(lgb_clf, max_num_features=30)
    booster = lgb_clf.booster_
    importance = booster.feature_importance(importance_type='split')
    feature_name = booster.feature_name()

    feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
    feature_importance.to_csv('feature_importance.csv',sep='\t',index=False)
    
    print('开始预测')
    y_prob = lgb_clf.predict_proba(test_x)

    test_data_id['prob'] = y_prob[:,1]
    print('输出成功')
    test_data_id.to_csv('submission_linxi_0724.csv',index=False, header=False)

predcit_cv()
# predcit_result()
