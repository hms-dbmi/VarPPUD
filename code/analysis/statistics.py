#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:23:51 2021

@author: rayin
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re

from pprint import pprint
from collections import Counter
from tableone import TableOne
from sdv.evaluation import evaluate
from sdv.metrics.relational import KSTestExtended
from sdv.metrics.tabular import CSTest, KSTest
from sdv.metrics.tabular import BNLikelihood
from sdv.metrics.tabular import LogisticDetection, SVCDetection
from sdv.metrics.tabular import BinaryAdaBoostClassifier

from feature_data_imputation import data_imputation


warnings.filterwarnings("ignore")

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")
sys.path.append(os.path.abspath("/Users/rayin/Google Drive/Harvard/5_data/UDN/work/code/"))

#import all the gene available data from 'variant_clean.csv'
case_gene_clean = pd.read_csv('data/processed/variant_clean.csv', index_col=0)
#case_gene_clean['\\12_Candidate variants\\03 Interpretation\\'].replace('pathogenic', 1, inplace=True)
#case_gene_clean['\\12_Candidate variants\\03 Interpretation\\'].replace('less_pathogenic', 0, inplace=True)
label = case_gene_clean['\\12_Candidate variants\\03 Interpretation\\'].reset_index()
label = label[label.columns[1]]

#Extract demographic information from 'case_gene_filter_labeled.csv'
case_gene_filter_labeled = pd.read_csv('data/processed/case_gene_update.csv', index_col=0)

case_demographics = []
for i in range(0, len(case_gene_clean)):
    for j in range(0, len(case_gene_filter_labeled)):
        if case_gene_clean['\\000_UDN ID\\'].iloc[i] ==  case_gene_filter_labeled['\\000_UDN ID\\'].iloc[j]:
            case_demographics.append(case_gene_filter_labeled.iloc[j])
            break
case_demographics = pd.DataFrame(case_demographics)
case_demographics = case_demographics.reset_index()
case_demographics = case_demographics.iloc[:,2:10]

patient_demographics = pd.concat([case_demographics, label], axis=1)
column_name = patient_demographics.columns
patient_demographics.rename(columns={column_name[0]: 'UDN ID', column_name[1]: 'Age at UDN Evaluation (in years)',
                                     column_name[2]: 'Age at symptom onset in years', column_name[3]: 'Current age in years',
                                     column_name[4]: 'Ethnicity', column_name[5]: 'Gender',
                                     column_name[6]: 'Race', column_name[7]: 'Primary symptom category reported by patient or caregiver',
                                     column_name[8]: 'Interpretation'}, inplace=True)
simple_variable = patient_demographics.columns


categorical = [simple_variable[4], simple_variable[5], simple_variable[6], simple_variable[7]]
groupby = simple_variable[8]
columns = ['Age at UDN Evaluation (in years)', 'Age at symptom onset in years', 'Current age in years', 'Ethnicity',
       'Gender', 'Race', 'Primary symptom category reported by patient or caregiver', 'Interpretation']
df = patient_demographics.iloc[:, patient_demographics.columns != simple_variable[0]]
mytable = TableOne(df, columns=columns, categorical=categorical, groupby=groupby)
#mytable.tableone.to_csv('result/patient_demographics.csv')


#statistics for features
#The value range of features (physicochemical properties)
polarity = [-0.06, -0.84, -0.48, -0.80, 1.36, -0.73, -0.77, -0.41, 0.49, 1.31, 1.21, -1.18, 1.27, 1.27, 0.0, -0.50, -0.27, 0.88, 0.33, 1.09]
net_charge = [0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
hydrophobicity = [0.36, -0.52, -0.90, -1.09, 0.70, -1.05, -0.83, -0.82, 0.16, 2.17, 1.18, -0.56, 1.21, 1.01, -0.06, -0.60, -1.20, 1.31, 1.05, 1.21]
normalized_vdw = [1.00, 6.13, 2.95, 2.78, 2.43, 3.95, 3.78, 0.00, 4.66, 4.00, 4.00, 4.77, 4.43, 5.89, 2.72, 1.60, 2.60, 8.08, 6.47, 3.00]
polarizability = [0.046, 0.291, 0.134, 0.105, 0.128, 0.180, 0.151, 0.000, 0.230, 0.186, 0.186, 0.219, 0.221, 0.290, 0.131, 0.062, 0.108, 0.409, 0.298, 0.140]
pK_COOH = [2.34, 1.18, 2.02, 2.01, 1.65, 2.17, 2.19, 2.34, 1.82, 2.36, 2.36, 2.18, 2.28, 1.83, 1.99, 2.21, 2.10, 2.38, 2.20, 2.32]
pK_NH2 = [9.69, 8.99, 8.80, 9.60, 8.35, 9.13, 9.67, 9.78, 9.17, 9.68, 9.60, 9.18, 9.21, 9.18, 10.64, 9.21, 9.10, 9.44, 9.11, 9.62]
hydration = [-1.0, 0.3, -0.7, -1.2, 2.1, -0.1, -0.7, 0.3, 1.1, 4.0, 2.0, -0.9, 1.8, 2.8, 0.4, -1.2, -0.5, 3.0, 2.1, 1.4]
molecular_weight = [89.09, 174.20, 132.12, 133.10, 121.15, 146.15, 147.13, 75.07, 155.16, 131.17, 131.17, 146.19, 149.21, 165.19, 
                    115.13, 105.09, 119.12, 204.24, 181.19, 117.15]
optical_rotation = [1.80, 12.50, -5.60, 5.05, -16.50, 6.30, 12.00, 0.00, -38.50, 12.40, -11.00, 14.60, -10.00, -34.50, -86.20,
                    -7.50, -28.00, -33.70, -10.00, 5.63]
secondary_structure = [1, 1, 3, 3, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 3, 3, 2, 2, 2, 2]
solvent_accessibility = [-1, 1, 1, 1, -1, 1, 1, -1, 0, -1, -1, 1, 0, -1, 0, 0, 0, -1, 0, -1]
free_energy_solution = [-0.368, -1.03, 0.0, 2.06, 4.53, 0.731, 1.77, -0.525, 0.0, 0.791, 1.07, 0.0, 0.656, 1.06, -2.24, -0.524, 0.0, 1.60, 4.91, 0.401]
number_of_hydrogen_bond = [0, 4, 2, 1, 0, 2, 1, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 0]
volumes_of_residues = [91.5, 196.1, 138.3, 135.2, 114.4, 156.4, 154.6, 67.5, 163.2, 162.6, 163.4, 162.5, 165.9, 198.8, 123.4, 102.0, 126.0, 209.8, 237.2, 138.4]
transfer_free_energy = [0.3, -1.4, -0.5, -0.6, 0.9, -0.7, -0.7, 0.3, -0.1, 0.7, 0.5, -1.8, 0.4, 0.5, -0.3, -0.1, -0.2, 0.3, -0.4, 0.6]
side_chain_interaction = [10.04, 6.18, 5.63, 5.76, 8.89, 5.41, 5.37, 7.99, 7.49, 8.7, 8.79, 4.40, 9.15, 7.98, 7.79, 7.08, 7.00, 8.07, 6.90, 8.88]
number_of_vertices = [2.00, 8.00, 5.00, 5.00, 3.00, 6.00, 6.00, 1.00, 7.00, 5.00, 5.00, 6.00, 5.00, 8.00, 4.00, 3.00, 4.00, 11.00, 9.00, 4.00]
number_of_edges = [1.00, 7.00, 4.00, 4.00, 2.00, 5.00, 5.00, 0.00, 6.00, 4.00, 4.00, 5.00, 4.00, 8.00, 4.00, 2.00, 3.00, 12.00, 9.00, 3.00]
eccentricity = [1.00, 8.120, 5.00, 5.17, 2.33, 5.860, 6.00, 0.00, 6.71, 3.25, 5.00, 7.00, 5.40, 7.00, 4.00, 1.670, 3.250, 11.10, 8.88, 3.25]
diameter = [1.00, 12.00, 6.00, 6.00, 3.00, 8.00, 8.00, 0.00, 9.00, 6.00, 6.00, 9.00, 7.00, 11.000, 4.000, 3.00, 4.00, 14.000, 13.000, 4.00]
atomic_number = [12.00, 45.00, 33.007, 34.00, 28.00, 39.00, 40.00, 7.00, 47.00, 30.0, 30.00, 37.00, 40.00, 48.00, 24.00, 22.00, 27.00, 68.00, 56.00, 24.00]

def feature_range(value):
    result = max(value) - min(value)
    print(result)

#feature_range(atomic_number)


#missing and mean (sd) information for all features
feature = pd.read_csv('data/feature/feature.csv', index_col=0)
feature_column = feature.columns.values

missing_num = []
presence = []
mean_value = []
sd_value = []
for i in range(0, feature.shape[1]):
    missing_count = np.isnan(feature[feature_column[i]]).sum()
    present_count = len(feature) - missing_count
    missing_num.append(missing_count)
    presence.append(round(present_count/len(feature), 3) * 100)
    mean_value.append(format(np.mean(feature[feature_column[i]]), '.3g'))
    sd_value.append(format(np.std(feature[feature_column[i]], ddof=1), '.3g'))



# #evaluate GAN-based synthetic data
# case_gene_update = pd.read_csv('data/processed/variant_clean.csv', index_col=0)
# case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('pathogenic', 1, inplace=True)
# case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('less_pathogenic', 0, inplace=True)
# label = case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].reset_index()
# label = label[label.columns[1]]
# feature = pd.read_csv('data/feature/feature_update_name.csv', index_col=0)
# real_sample = pd.concat([feature, label], axis=1)
# real_sample.rename(columns={"\\12_Candidate variants\\03 Interpretation\\": "label"}, inplace=True)

# # feature_imputation = data_imputation(feature, 'MICE') 
# # real = pd.concat([feature_imputation, label], axis=1)

# syn_test_sample = pd.read_csv('data/synthetic/syn_test.csv', index_col=0)
# syn_test_sample = pd.read_csv('data/synthetic/syn_test_raw.csv', index_col=0)
# syn_test_sample = pd.read_csv('data/synthetic/syn_data_raw.csv', index_col=0)


# evaluate(syn_test_sample, real_sample, aggregate=False)
# CSTest.compute(real_sample, syn_test_sample)
# KSTest.compute(real_sample, syn_test_sample)
# evaluation_test = evaluate(real_sample, syn_test_sample, metrics=['CSTest', 'KSTest'], aggregate=False)

# BNLikelihood.compute(real_sample.fillna(0), syn_test_sample.fillna(0))
# LogisticDetection.compute(real_sample, syn_test_sample)
# SVCDetection.compute(real_sample, syn_test_sample)

# BinaryAdaBoostClassifier.compute(real_sample, syn_test_sample, target='label')








