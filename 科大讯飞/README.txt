1.首先在code文件夹里新建data文件夹，放入初赛训练数据集round1_iflyad_train.txt、复赛训练数据集round2_iflyad_train.txt、复赛测试数据集round2_iflyad_test_feature.txt；

2.运行base_feature.py文件，生成基本特征数据集base_train_csr.npz和base_predict_csr.npz，放在data文件夹中；

3.运行lgb_model_1.py文件，训练第一个模型，生成文件sub1.csv,放在data文件下；

4.运行lgb_model_2.py文件，训练第二个模型，生成文件sub2.csv,放在data文件下；

5.依次运行s_0.py,s_1.py,s_2.py,s_3.py,main.py,s_5.py，生成文件sub3.csv,放在data文件下；

6.运行final_result.py文件，生成最终提交文件sub.csv,放在data文件下；

7.sub.csv为最终提交文件。


