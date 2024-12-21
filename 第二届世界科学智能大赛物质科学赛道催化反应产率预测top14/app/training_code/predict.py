# 导入库
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
import os
from catboost import CatBoostRegressor
# from  deepforest import classdeepforest
import random
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import KFold
from rdkit.Chem import QED
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
    #smi_vec_map[''] = np.zeros(2048+167+6)
    vec_lst = [smi_vec_map[smi] for smi in smi_lst]
    return np.array(vec_lst)

def feature_processing(train_df,aug=None):
    columns = ["Reactant1", "Reactant2", "Product",  "Additive", "Solvent"]
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

dataset_dir = 'tcdata'
test_df = pd.read_csv(f'/{dataset_dir}/input/round2_test_data.csv')
#test_df = pd.read_csv("data/round1_rxntype_test_data.csv")

test_x = feature_processing(test_df)

# 加载模型
with open('5k_stacking_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 预测\推理
test_pred = loaded_model.predict(test_x)
print(test_pred)

ans_str_lst = ['rxnid,Yield']
for idx, y in enumerate(test_pred):
    ans_str_lst.append(f'test{idx + 1},{y:.4f}')
os.makedirs(os.path.dirname("/app/output/submit.txt"),exist_ok=True)
with open('/app/output/submit.txt', 'w') as fw:
    fw.writelines('\n'.join(ans_str_lst))
