# -*- coding:utf-8 -*-
import pandas as pd

pf1 = pd.read_csv("round1_train_data.csv")
pf2 = pd.read_csv("round2_train_data.csv")

pf1["rxntype"] = len(pf1)*[1]

pf = pd.concat([pf1,pf2])
pf.to_csv("combine_train_data.csv",index=None)

pf1 = pd.read_csv("round1_test_data.csv")
pf1["rxntype"] = len(pf1)*[1]
pf1.to_csv("round1_rxntype_test_data.csv",index=None)