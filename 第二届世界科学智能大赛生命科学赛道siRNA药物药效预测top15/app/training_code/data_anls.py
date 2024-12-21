import pandas as pd
import numpy as np

import random
def sample(data_dict):
    """
    in: {"A":135,"B":200} / {24.0: 21984, 48.0: 2716}
    out: A/B
    """
    values = sum(data_dict.values())
    prob = [round(i/values,4)for i in data_dict.values()]
    keys = list(data_dict.keys())

    x=random.uniform(0,1)
    probability=0.0
    for item,item_probability in zip(keys,prob):
        probability+=item_probability
        if x < probability: break
    return item

if __name__ =="__main__":
    train_data = pd.read_csv('../data/train_data.csv')
    print(train_data.isnull().sum())
    train_data["gene_target_symbol_name"] = train_data["gene_target_symbol_name"].apply(
        lambda x: "unknown" if x is np.nan else x)
    train_data["gene_target_ncbi_id"] = train_data["gene_target_ncbi_id"].apply(
        lambda x: "unknown" if x is np.nan else x)
    gene_target_species = train_data["gene_target_species"].value_counts().to_dict()
    train_data["gene_target_species"] = train_data["gene_target_species"].apply(
        lambda x: sample(gene_target_species) if x is np.nan else x)
    train_data["cell_line_donor"] = train_data["cell_line_donor"].apply(lambda x: "unknown" if x is np.nan else x)
    Transfection_method = train_data["Transfection_method"].value_counts().to_dict()
    Duration_after_transfection_h = train_data["Duration_after_transfection_h"].value_counts().to_dict()
    train_data["Transfection_method"] = train_data["Transfection_method"].apply(
        lambda x: sample(Transfection_method) if x is np.nan else x)
    train_data["Duration_after_transfection_h"]=train_data["Duration_after_transfection_h"].apply(lambda x: sample(Duration_after_transfection_h) if np.isnan(x) else x)

    print(train_data.isnull().sum())
    train_data.to_csv('../data/anl_train_data.csv', index=None)