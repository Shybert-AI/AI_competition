# 导入库
import os
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor,BaggingRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from  deepforest import classdeepforest
import random
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import KFold

# https://rxn4chemistry.github.io/rxnfp/
import sys
sys.path.append("rxnfp-master")

def randomize_smiles(mol, random_type="rotated", isomericSmiles=True):
    """
    From: https://github.com/undeadpixel/reinvent-randomized and https://github.com/GLambard/SMILES-X
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted, rotated) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    #mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=isomericSmiles)
    elif random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=isomericSmiles)
    elif random_type == 'rotated':
        n_atoms = mol.GetNumAtoms()
        rotation_index = random.randint(0, n_atoms-1)
        atoms = list(range(n_atoms))
        new_atoms_order = (atoms[rotation_index%len(atoms):]+atoms[:rotation_index%len(atoms)])
        rotated_mol = Chem.RenumberAtoms(mol,new_atoms_order)
        return Chem.MolToSmiles(rotated_mol, canonical=False, isomericSmiles=isomericSmiles)
    raise ValueError("Type '{}' is not valid".format(random_type))


def mfgen(mol,aug=None, nBits=2048, radius=2):
    if aug:
        index = random.sample(["restricted","rotated"],1)
        smiles = randomize_smiles(mol,random_type=index[0])
        mol = Chem.MolFromSmiles(smiles)
    # 返回分子的位向量形式的Morgan fingerprint
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    #fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius+1, nBits=nBits)
    fp3 = AllChem.GetMACCSKeysFingerprint(mol)
    #return np.array(list(map(int, list(fp.ToBitString()))))
    return np.concatenate([np.array(list(map(int, list(fp.ToBitString())))),np.array(list(map(int, list(fp3.ToBitString()))))])


def feature_strackt(mol):
    # 使用RDKit来计算2D和3D分子描述符。
    # http://www.360doc.com/content/22/0419/11/79325180_1027224666.shtml
    from rdkit.Chem import rdMolDescriptors,Descriptors
    feat1 = Descriptors.ExactMolWt(mol)
    feat2 = Chem.rdMolDescriptors.CalcTPSA(mol) #Topological Polar Surface Area
    feat3 = Descriptors.NumRotatableBonds (mol) #Number of rotable bonds
    feat4 = Descriptors.NumHDonors(mol) #Number of H bond donors
    feat5 = Descriptors.NumHAcceptors(mol) #Number of H bond acceptors
    feat6 = Descriptors.MolLogP(mol) #LogP
    return np.array([feat1,feat2,feat3,feat4,feat5,feat6])

# 加载数据
def vec_cpd_lst(smi_lst,aug):
    smi_set = list(set(smi_lst))
    smi_vec_map = {}
    for smi in tqdm(smi_set):  # tqdm：显示进度条
        mol = Chem.MolFromSmiles(smi)
        smi_vec_map[smi] = np.concatenate([mfgen(mol,aug),feature_strackt(mol)])  # 2048 + 6
        #smi_vec_map[smi] = np.concatenate([mfgen(mol,aug),np.array(_postion_Feature.features_postion(smi)[0])])
        #smi_vec_map[smi] = mfgen(mol, aug)

    #smi_vec_map[''] = np.zeros(2151)
    smi_vec_map[''] = np.zeros(2048+167+6)
    vec_lst = [smi_vec_map[smi] for smi in smi_lst]
    return np.array(vec_lst)

def feature_processing(train_df,aug=None):
    columns = ["Reactant1", "Reactant2", "Product",  "Additive", "Solvent"] #"rxntype"
    feature = []
    for i in columns:
        train_sol_smi = train_df[i].to_list()
        train_sol_fp = vec_cpd_lst(train_sol_smi,aug)
        # 添加rxntype的特征
        train_sol_fp = np.hstack((train_sol_fp, np.array(train_df["rxntype"].tolist()).reshape(-1, 1)))
        feature.append(train_sol_fp)

    train_x = np.mean(feature, axis=0)
    #train_x = np.hstack(feature)

    return train_x

def combine_features(row):
    stt = list(set([row['Reactant1'],row['Reactant2'],row['Product'],row['Additive'],row['Solvent']]))
    return " ".join(stt)
    #return ' '.join([str(row['Reactant1']), str(row['Reactant2']), str(row['Product']), str(row['Additive']), str(row['Solvent'])])



def combine_features1(row):
    ms = [Chem.MolFromSmiles(row['Reactant1']), Chem.MolFromSmiles(row['Reactant2'])]
    fps = [Chem.RDKFingerprint(x) for x in ms]
    Similar =DataStructs.TanimotoSimilarity(fps[0], fps[1])
    return Similar
def combine_features2(row):
    ms = [Chem.MolFromSmiles(row['Reactant1']), Chem.MolFromSmiles(row['Reactant2'])]
    fps = [Chem.RDKFingerprint(x) for x in ms]
    Similar = DataStructs.TanimotoSimilarity(fps[0], fps[1])
    return Similar


dataset_dir = '../data'  

train_df = pd.read_csv(f'{dataset_dir}/combine_train_data.csv')
test_df = pd.read_csv(f'{dataset_dir}/round1_rxntype_test_data.csv')


print(f'Training set size: {len(train_df)}, test set size: {len(test_df)}')

# columns = [
#     'Reactant1', 'Reactant2', 'Product', 'Additive', 'Solvent',"rxntype"
# ]

train_x = feature_processing(train_df)
train_y = train_df['Yield'].to_numpy()
test_x = feature_processing(test_df)

if False:
    train_x_aug = feature_processing(train_df, aug=True)
    train_x = np.concatenate([train_x, train_x_aug])
    train_y = np.concatenate([train_y, train_y])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 定义一级模型
#rf_model = RandomForestRegressor(n_estimators=1000, max_depth=None, max_features='sqrt', min_samples_leaf=1,min_samples_split=5, n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=-1,criterion="friedman_mse")
gb_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=20, max_features='sqrt',
                                     min_samples_leaf=1, min_samples_split=6)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=20, min_child_weight=0.1, subsample=1,
                         colsample_bytree=0.4, n_jobs=-1)

lgb_model = LGBMRegressor(boosting_type='gbdt',objective='huber', num_leaves=31, max_depth=-1,
learning_rate=0.1, n_estimators=3000, max_bin=255, subsample_for_bin=400000,
min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.7,
subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=0.0, random_state=None,
n_jobs=-1)

# 定义堆叠模型
stacking_model = StackingRegressor(
    estimators=[('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model),("lgb",lgb_model)],
    final_estimator=LinearRegression()
)

# ('rf', rf_model)   Validation MSE: 0.03095357 Validation R2: 0.39878971
# ('gb', gb_model)   Validation MSE: 0.03076948 Validation R2: 0.40236532
# ('xgb', xgb_model)  Validation MSE: 0.03076972 Validation R2: 0.40236064
# ("lgb",lgb_model)  
# n_estimators=300 Validation MSE: 0.03695800 Validation R2: 0.28216591   
# n_estimators=2000 Validation MSE: 0.03261208 Validation R2: 0.36657650
# n_estimators=2000  learning_rate=0.1 Validation MSE: 0.03212943 Validation R2: 0.37595098
# n_estimators=3000  learning_rate=0.1 Validation MSE: 0.03181048 Validation R2: 0.38214603
# n_estimators=3000  learning_rate=0.1 Validation MSE: 0.03181048 Validation R2: 0.38214603

#from sklearn.model_selection import train_test_split, GridSearchCV
#gridParams = {
#'reg_alpha': [0.1,0.05],
#'reg_lambda': [0.1,0.05],
#    'learning_rate': [0.1,0.05,0.01],
#}
##lgb_model {'colsample_bytree': 0.7, 'max_depth': -1, 'min_child_weight': 0.1, 'subsample': 1.0}
#reg_gridsearch = GridSearchCV(lgb_model, gridParams, cv=5, scoring='r2', n_jobs=-1) 
#reg_gridsearch.fit(X_train, y_train,eval_set=(X_val,y_val))
#print(reg_gridsearch.best_params_ )
#
#import time
#print(111111111111111)
#time.sleep(100000)

# 训练堆叠模型
stacking_model.fit(X_train, y_train)
# 在测试集上进行预测

y_val_pred = stacking_model.predict(X_val)
print(f'Validation MSE: {mean_squared_error(y_val, y_val_pred):.8f}')
print(f'Validation R2: {r2_score(y_val, y_val_pred):.8f}')

# # 使用整个训练集训练堆叠模型
stacking_model.fit(train_x, train_y)

# 保存模型
with open('./5k_stacking_model.pkl', 'wb') as file:
    pickle.dump(stacking_model, file)

# 加载模型
with open('5k_stacking_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 预测\推理
test_pred = loaded_model.predict(test_x)

ans_str_lst = ['rxnid,Yield']
for idx, y in enumerate(test_pred):
    ans_str_lst.append(f'test{idx + 1},{y:.4f}')
with open('./5k_baseline_submit.txt', 'w') as fw:
    fw.writelines('\n'.join(ans_str_lst))
