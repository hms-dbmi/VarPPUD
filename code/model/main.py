#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:12:46 2021

@author: rayin
"""


import os, sys
import numpy as np
import pandas as pd
import torch
import warnings
import torchvision.models as models
import matplotlib.pyplot as plt
import shap
import sympy as sp

sys.path.append(os.path.abspath("/Users/rayin/Google Drive/Harvard/2_paper/2021/Gene pathogenicity prediction/PNAS/github/code/"))

from model import multi_experiment
from model import draw_roc_curve
from model import patient_variable_influence
from model import draw_roc_train_test_split
from model import draw_roc_syn_test
from model import draw_pr_roc
from feature_data_imputation import data_imputation

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import ensemble

warnings.filterwarnings("ignore")

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")


#import all the gene available data
case_gene_update = pd.read_csv('data/processed/variant_clean.csv', index_col=0)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('pathogenic', 1, inplace=True)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('less_pathogenic', 0, inplace=True)
label = case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].reset_index()
label = label[label.columns[1]]
feature = pd.read_csv('data/feature/feature.csv', index_col=0)
feature_imputation = data_imputation(feature, 'MICE') 
patient_level_variable_onehot = pd.read_csv('data/feature/patient_level_variable_onehot.csv', index_col=0)
syn_test_sample = pd.read_csv('data/synthetic/syn_test.csv', index_col=0)


#calculating prediction performance changes adding demographic information. 'raw': original features. 
patient_variable = ['raw', 'age_symptom_onset', 'current_age', 'ethnicity', 'gender', 'race', 'all']
for i in range(0, len(patient_variable)):
    patient_variable_influence('rf', feature_imputation, label, patient_level_variable_onehot, patient_variable[i])


#running with proposed model  
multi_experiment('rf', 5, feature, label)

#plotting roc curve using synthetic data
draw_roc_syn_test(feature_imputation, label, syn_test_sample, 'rf')  

#plotting pr curve using synthetic data
draw_pr_roc(feature_imputation, label, syn_test_sample)

# draw_roc_train_test_split(feature_imputation, label, syn_test_sample, 'rf')
# method = ['lr', 'rf', 'xgboost']
# for i in range(0, len(method)):
#     draw_roc_train_test_split(feature_imputation, label, method[I])
































