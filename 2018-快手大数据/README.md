# 2018中国高校计算机大赛——大数据挑战赛

## 比赛官网：
https://www.kesci.com/apps/home/competition/5ab8c36a8643e33f5138cba4/content

## 1. 大佬开源
[2018中国高校计算机大赛--大数据挑战赛 1-lambda](https://github.com/drop-out/RNN-Active-User-Forecast)  
[2018中国高校计算机大赛--大数据挑战赛 4-lctry](https://github.com/chantcalf/2018-Rank4-)  
[2018中国高校计算机大赛--大数据挑战赛 13-蛇](https://github.com/luoda888/2018-KUAISHOU-TSINGHUA-Top13-Solutions)  
[2018中国高校计算机大赛--大数据挑战赛 13-RNN-小狼](https://github.com/totoruo/KuaiShou2018-RANK13-RNN)  
[2018中国高校计算机大赛--大数据挑战赛 20](https://github.com/bigzhao/Kuaishou_2018_rank20th)  
[复赛B榜20+ 单LGB 0.91228](https://github.com/hellobilllee/ActiveUserPrediction/)  
[复赛B榜20+ 单LGB 0.9121](https://github.com/ZesenChen/kuaishou2018)  
[2018中国高校计算机大赛--大数据挑战赛-30](https://github.com/senmonster/2018-KUAISHOU-TSINGHUA-RANK30-Solutions)  
[2018中国高校计算机大赛--大数据挑战赛 35-chizhu](https://github.com/chizhu/kuaishou2018)  
[2018中国高校计算机大赛--大数据挑战赛 43-Xgboost+LSTM](https://github.com/xwsvincent/2018-KuaiShouAPP-ActiveUser-Prediction)  
[2018中国高校计算机大数据挑战赛-快手活跃用户预测](https://blog.csdn.net/u014775977/article/details/81705271)  
[guoday-Github](https://github.com/guoday) [BDC_2018](https://github.com/guoday/BDC_2018)  

## 2. 经验帖：

### 2.1 论坛分享：
[整体框架-得分0.811](https://www.kesci.com/apps/home/competition/forum/5b10d0ce79433a0277555bab)  
[Baseline0.79,No Xgboost](https://www.kesci.com/apps/home/competition/forum/5b08ec32e811fa77fb091433)  
[0.82分享（非周星星)-Snake](https://www.kesci.com/apps/home/competition/forum/5b1a6e69fe8bc06aa3aafec0)  

### 2.2 第一周 周星星分享：
[周星星分享(Excuses)](https://www.kesci.com/apps/home/competition/forum/5b10f29179433a0277558229)  
[周星星分享(小胖胖)](https://www.kesci.com/apps/home/competition/forum/5b124aae9cb0850279d46b84)  
[周星星分享(chantcalf)](https://www.kesci.com/apps/home/competition/forum/5b10ee5879433a0277557cdd)

### 2.3 第二周 周星星分享：
[周星星分享(Thanos)](https://www.kesci.com/apps/home/competition/forum/5b1a32fbfe8bc06aa3aab6bb)  
[周星星分享(光华)](https://www.kesci.com/apps/home/competition/forum/5b1a5899fe8bc06aa3aaeaa8)  
[周星星分享(ChenYiFaer)](https://www.kesci.com/apps/home/competition/forum/5b1a6084fe8bc06aa3aaf1f9)

### 2.4 第三周 周星星分享
[周星星分享(Derec)](https://www.kesci.com/apps/home/competition/forum/5b2cb519f110337467b28fbe)  
[周星星分享(丹尼韦德)](https://www.kesci.com/apps/home/competition/forum/5b2cd0b8f110337467b2b3f4)  
[周星星分享(区块链第一人)](https://www.kesci.com/apps/home/competition/forum/5b2c9d23f110337467b26d25)

### 2.5 复赛 第一次周周星分享
[周星星分享（lambda）](https://www.kesci.com/home/competition/forum/5b51fddffc7e90001032e34d)  
[周星星分享（柠檬没我萌）](https://www.kesci.com/home/competition/forum/5b5569c5fc7e9000103932bd)  
[周星星分享（可爱多）](https://www.kesci.com/home/competition/forum/5b52f8d5fc7e900010347c08)  

### 2.6 复赛 第二次周周星分享
[周星星分享（淡淡的丹尼）](https://www.kesci.com/home/competition/forum/5b5acaa2a711e6001098d9ee)  
[周星星分享（lindada）](https://www.kesci.com/home/competition/forum/5b5ad22da711e6001098f045)  
[周星星分享（lctry）](https://www.kesci.com/home/competition/forum/5b5abf73a711e6001098ba48)  

### 2.7 复赛 第三次周周星分享
[周星星分享（神偷奶爸）](https://www.kesci.com/home/competition/forum/5b6452646a25e70011edbc83)  
[周星星分享（扫地僧）](https://www.kesci.com/home/competition/forum/5b6446cd6a25e70011ed99d1)  
[周星星分享（X-Driver）](https://www.kesci.com/home/competition/forum/5b64732c6a25e70011ee23a7)  

## 3.个人代码
* baseline1：第一版本，最初的特征，没有模型融合等；
* baseline2：第二版，加了很多特征，上分明显，上了30+名；
* baseline_CV：新版CV，相当于一种模型融合方式吧，2个万分点大概；
* baseline_pca：尝试PCA添加特征，线下有提升，奈何，线上没机会尝试；
* baseline_nn:神经网络，简单全连接，最后两天才写出来，比lgb低5个千；
