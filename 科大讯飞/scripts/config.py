# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:27:20 2018

@author: SY
"""

lable_name = 'click'
RANDOM_STATE = 575+3+5 # 3

lst = ['adid', 'advert_id', 'advert_industry_inner', 'advert_name', 'app_cate_id', 'app_id', 
		'app_paid', 'campaign_id', 'carrier', 'city', 'creative_has_deeplink', 'creative_height', 
		'creative_id', 'creative_is_download', 'creative_is_js', 'creative_is_jump', 'creative_is_voicead', 
		'creative_tp_dnf', 'creative_type', 'creative_width', 'devtype', 'f_channel', 'inner_slot_id', 
		'make', 'model', 'nnt', 'orderid', 'os', 'osv', 'province', 'user_tags', 'hour', 
		'advert_industry_inner_0', 'advert_industry_inner_1', 'inner_slot_id_0', 'OSV', 'OSV_0', 'user_adid', 
		'make_area_mean', 'advert_industry_inner_1_ts_std', 'area', 'uid_adid_nunique', 'creative_tp_dnf_hour_mean', 
		'uid_cnt', 'uid_creative_type_nunique', 'advert_name_height_mean', 'uid_inner_slot_id_nunique', 
		'creative_tp_dnf_width_mean', 'advert_id_day_std', 'advert_id_ts_max', 'make_height_std', 
		'advert_industry_inner_1_hour_std', 'advert_industry_inner_day_ratio', 'model_day_std', 
		'uid_app_cate_id_cnt', 'creative_tp_dnf_area_std', 'advert_industry_inner_1_hour_mean', 
		'model-', 'creative_type_ts_mean', 'adid_day_std', 'model_Redmi', 'advert_name_day_ratio', 
		'adid_day_ratio', 'uid_advert_industry_inner_1_cnt_ratio', 'creative_type_width_mean', 
		'uid_advert_id_cnt', 'model_hour_mean', 'adid_height_mean', 'model_height_std', 
		'uid_creative_tp_dnf_cnt_ratio', 'inner_slot_id_day_std', 'creative_tp_dnf_height_std', 
		'make_day_ratio', 'inner_slot_id_ts_mean', 'adid_hour_std', 'model_width_std', 
		'uid_advert_id_cnt_ratio', 'model_hour_std', 'model_day_ratio', 'osv_len', 'make_ts_mean', 
		'uid_creative_type_cnt_ratio', 'osv_day_ratio', 'make_ts_max', 'uid_adid_cnt_ratio', 
		'uid_inner_slot_id_cnt_ratio', 'advert_id_ts_std', 'inner_slot_id_area_std', 'adid_area_mean', 
		'make_ts_std', 'app_id_height_mean', 'uid_advert_id_nunique', 'app_cate_id_hour_mean', 
		'osv_hour_mean', 'make_height_mean', 'model_width_mean', 'app_id_height_std', 
		'creative_type_day_std', 'app_id_area_std', 'advert_id_ts_mean', 'adid_width_mean', 
		'creative_tp_dnf_width_std', 'uid_inner_slot_id_cnt', 'inner_slot_id_hour_std', 
		'inner_slot_id_width_std', 'advert_name_area_std', 'creative_tp_dnf_day_std', 'osv_hour_std', 
		'advert_name_width_std', 'osv_area_mean', 'app_id_width_std', 'inner_slot_id_hour_mean', 
		'model_Plus', 'uid_advert_industry_inner_1_nunique', 'creative_tp_dnf_day_ratio', 'adid_width_std', 
		'uid_creative_type_cnt', 'adid_hour_mean', 'osv_width_mean', 'osv_day_std', 'app_id_hour_mean', 
		'advert_name_width_mean', 'app_cate_id_hour_std', 'uid_creative_tp_dnf_nunique', 
		'advert_industry_inner_1_ts_max', 'make_width_mean', 'adid_ts_max', 'adid_height_std', 
		'inner_slot_id_height_mean', 'uid_app_cate_id_nunique', 'model,', 
		'osv_width_std', 'make_new', 'model+', 'adid_ts_std', 'app_id_day_ratio', 'model_height_mean', 
		'uid_app_id_nunique', 'advert_id_hour_mean', 'inner_slot_id_width_mean', 'uid_day_nunique', 
		'make_hour_std', 'creative_type_day_ratio', 'uid_app_id_cnt', 'advert_name_height_std', 
		'advert_industry_inner_1_ts_mean', 'app_id_area_mean', 'creative_type_height_std', 
		'uid_creative_tp_dnf_cnt', 'app_id_hour_std', 'osv_height_std', 
		'creative_type_area_std', 'uid_app_cate_id_cnt_ratio', 'creative_type_hour_std', 
		'advert_name_area_mean', 'creative_tp_dnf_hour_std', 'inner_slot_id_height_std', 
		'creative_type_width_std', 'osv_0', 'make_width_std', 'creative_tp_dnf_ts_max', 
		'advert_industry_inner_1_day_std', 'creative_type_area_mean', 'inner_slot_id_area_mean', 
		'osv_area_std', 'advert_name_hour_mean', 'advert_name_hour_std', 'uid_app_id_cnt_ratio', 
		'app_cate_id_day_std', 'model_s', 'adid_ts_mean', 'uid_advert_industry_inner_1_cnt', 
		'inner_slot_id_ts_std', 'make_area_std', 'osv_height_mean', 'inner_slot_id_day_ratio', 
		'uid_adid_cnt', 'creative_type_height_mean', 'w_h_ratio', 'inner_slot_id_ts_max', 
		'creative_type_hour_mean', 'advert_id_hour_std', 'adid_area_std', 'model_area_std', 
		'creative_tp_dnf_area_mean', 'creative_type_ts_max', 'creative_tp_dnf_ts_std', 
		'uid_day_cnt', 'make_day_std', 'app_id_day_std', 'make_hour_mean', 'model_area_mean', 
		'creative_type_ts_std', 'model%', 'app_id_width_mean', 'model ', 'creative_tp_dnf_ts_mean', 
		'creative_tp_dnf_height_mean', 'advert_name_day_std', 'ts_diff']

stacking_dict = {}


stacking_dict[0] = ['adid', 'advert_id', 'advert_industry_inner', 'advert_name', 
                 'app_cate_id', 'app_id', 'app_paid', 'campaign_id', 'carrier', 'city', 
                 'creative_has_deeplink', 'creative_height', 'creative_id', 'creative_is_download', 
                 'creative_is_js', 'creative_is_jump', 'creative_is_voicead', 'creative_tp_dnf', 
                 'creative_type', 'creative_width', 'devtype', 'f_channel', 'inner_slot_id', 
                 'make', 'model', 'nnt', 'orderid', 'os', 'osv', 'province', 'user_tags', 'hour', 
                 'advert_industry_inner_0', 'advert_industry_inner_1', 'inner_slot_id_0', 'OSV', 
                 'OSV_0', 'user_adid']

t = 40
import random
randomlist = random.sample(lst, len(lst))
for i, ix in enumerate(range(0, len(lst), t), 1):
    stacking_dict[i] = randomlist[ix:min(ix+t, len(lst))]
randomlist = random.sample(lst, len(lst))
for i, ix in enumerate(range(0, len(lst), t), 10+1):
    stacking_dict[i] = randomlist[ix:min(ix+t, len(lst))]
randomlist = random.sample(lst, len(lst))
for i, ix in enumerate(range(0, len(lst), t), 10*2+1):
    stacking_dict[i] = randomlist[ix:min(ix+t, len(lst))]


cv_feats = ['user_tags']
onehot_feats = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 
                'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_id', 
                'creative_type', 'creative_tp_dnf', 'creative_has_deeplink', 
                'creative_is_jump', 'creative_is_download', 'app_cate_id', 
                'f_channel', 'app_id', 'inner_slot_id', 'city', 'carrier', 
                'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

type_dict = {}
for item in cv_feats:
    type_dict[item] = 'cv'
for item in onehot_feats:
    type_dict[item] = 'onehot'
    
