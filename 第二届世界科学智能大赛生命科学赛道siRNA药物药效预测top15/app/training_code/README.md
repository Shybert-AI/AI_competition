# 1.目录结构
  README.md
│
├─code
│  │  baseline.py    # 训练脚本  
│  │  data_anls.py   # 数据预处理  
│  │  main.py        # 主程序  
│  │  predict.py     # 测试脚本  
├─data
└─submit 

# 2.预装环境 python 3.12
easydict==1.13  
matplotlib==3.8.4  
numpy==1.24.3  
pandas==2.0.0  
rich==13.7.1  
scikit_learn==1.4.2  
torch==2.2.2  
tqdm==4.66.4  

# 3.比赛方案
&nbsp;&nbsp;&nbsp;&nbsp;本次比赛方案采用基因分词器GenomicTokenizer，对基因序列进行编码，然后在采用gru进行模型训练，预测mRNA_remaining_pct的值，此次选择的特征包含：
columns =  [
        #'id',
        #'publication_id',
        'gene_target_symbol_name',
        'gene_target_ncbi_id',
        'gene_target_species',
        'siRNA_duplex_id',
        'siRNA_sense_seq',
        'siRNA_antisense_seq',
        'cell_line_donor',
        'siRNA_concentration',
        #'concentration_unit',
        'Transfection_method',
        'Duration_after_transfection_h',
        'modified_siRNA_sense_seq',
        'modified_siRNA_antisense_seq',
        'modified_siRNA_sense_seq_list',
        'modified_siRNA_antisense_seq_list',
        "siRNA_sense_seq_gc",
        "siRNA_antisense_seq_gc"
        #'gene_target_seq',
        #'mRNA_remaining_pct'
    ]；模型的上分策略包括增加特征的选择、不同epoch模型融合、siRNA序列GC含量的计算和全量数据训练模型。
# 4.数据预处理
cd app/training_code && python data_anls.py
# 5.模型训练
cd app/training_code && python baselinev1.py
# 6.模型推理
cd app/training_code && python predict.py
# 7.提交预测结果文件，预测结果文件/app/submit.csv


