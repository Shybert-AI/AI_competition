# 1.目录结构
│  REAMDE.md  
│
├─training_code  
│      5Kmodefusion_0.4539.py          #模型训练 
│      csv_hebing.py                   #数据预处理脚本
       predict.py                      #推理脚本
│      requirements.txt  
│
├─data  
│      combine_train_data.csv        # 训练数据  
│      round1_rxntype_test_data.csv  # 测试数据  
│
└─submit  

# 2.预装环境 python 3.12
lightgbm==4.5.0  
numpy==1.26.4  
pandas==2.2.2  
rdkit==2024.3.3  
scikit_learn==1.4.2  
tqdm==4.66.4  
xgboost==2.1.1  

# 3.比赛方案
&nbsp;&nbsp;&nbsp;&nbsp;本次比赛方案采用rdkit替换化学式smiles的摩尔特征，采用机器学习的方式进行RandomForestRegressor、GradientBoostingRegressor、XGBRegressor和
LGBMRegressor的回归模型预测催化反应效率。模型的上分策略包括增加特征的列数、增加融合模型的数量、全量funture已训练的模型、5组特征平均改成堆叠、KFold和采用数据增强进行训练。

# 4.数据预处理
   将初赛和复赛的数据合并，并将初赛的测试数据加rxntype字段
# 4.模型训练及推理
cd /app/training_code && 5Kmodefusion_0.4539.py

# 5.提交预测结果文件，预测结果文件在/app/output/submit.txt
cd /app/training_code && predict.py


