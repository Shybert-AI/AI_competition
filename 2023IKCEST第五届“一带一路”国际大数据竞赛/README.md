# 训练代码文件   
## 1.目录结构 
```
.
├── ernie-m-base                                                                     # ernie模型离线加载目录  
├── main_warmup2_增强_ocr_频域特征v2_融合_epoch4_100_0.84122-final.ipynb              # 训练代码  
├── model                                                                            # 最优模型保存路径  
├── paddlenlp                                                                        # paddlenlp-2.4.2的离线版本，paddlenlp-2.5.1后,ernie模型初始化会丢失部分权重  
├── paddlenlp-2.4.2.dist-info
├── PaddleOCR-2.6.0                                                                  # PaddleOCR-2.6.0的离线版本,安装采用jupyter左侧的套件安装  
├── predict.py                                                                       # 推理脚本       
├── predict_system.py                                                                # 修改的tools/infer/predict_system.py脚本，用于进行图片的OCR识别   
├── README.md                                                                        # 说明文档  
├── requirements.txt                                                                 # 环境依赖  
├── result.csv                                                                       # 结果文件  
├── submission.zip                                                                   # submission.zip为榜单上最终成绩文件
└── 团队信息.txt
```
## 2.环境配置 
&nbsp;&nbsp;环境采用的是paddle框架进行训练的,paddlenlp-2.4.2(paddlenlp-2.5.1之后,ernie模型初始化不完整，会丢失部分权重)，显卡配置V100 32G,安装包依赖见requirements.txt
## 3.环境依赖包  
requirements.txt
## 4.数据输入  
在main_warmup2_增强_ocr_频域特征v2_融合_epoch4_100_0.84122-final.ipynb的4.2前读取数据修改path的路径为训练数据，如只需要修改path  
#读取数据
path = "/home/aistudio/queries_dataset_merge"
## 5.结果输出  
分类结果输出的在训练代码定义开始训练的32行，为
# 输出类别的概率
probs = F.softmax(probs)
## 6.方法介绍  
系统的设计是先采用PaddleOCR-2.6.0对训练集、测试集、验证集的img进行OCR识别，将识别的文件结果保存为同名的txt文件，用于在定义dataloader时进行加载。然后设计了将不同模态的特征采用Cross-Attention进行交互融合，最后将早期特征、交互特征和晚期特征进行融合，最终完成对基于多模态多特征融合的谣言稽查模型。训练的文本特征提取取采用的是ernie-m-base，图像特征提取器采用的是resnext101_64x4d。
训练代码采用的是main_warmup2_增强_ocr_频域特征v2_融合_epoch4_100_0.84122-final.ipynb, 只需要修改读取数据的path
## 7.推理文件  
submission.zip为榜单上最终成绩文件
