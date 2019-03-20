# -*- coding: utf-8 -*-
"""
@author: taopeng
"""

import pandas as pd

data1 = pd.read_csv('./data/sub1.csv')
data2 = pd.read_csv('./data/sub2.csv')
data3 = pd.read_csv('./data/sub3.csv')
data1 = data1.sort_values("instance_id")
data2 = data2.sort_values("instance_id")
data3 = data3.sort_values("instance_id")
data1['predicted_score'] =(data1['predicted_score']+data2['predicted_score']+data2['predicted_score'])/3
data1.to_csv('./data/sub.csv',index=False)
