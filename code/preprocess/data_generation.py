#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:19:39 2021

@author: rayin
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import warnings
import random
import torchvision.models as models

from sdv.tabular import CTGAN
from sdv.evaluation import evaluate
from sdv.metrics.tabular import CSTest, KSTest
from sdv.metrics.tabular import MulticlassDecisionTreeClassifier
from sdv.metrics.tabular import LogisticDetection, SVCDetection
from ctgan import CTGANSynthesizer
from feature_data_imputation import data_imputation
from sdv.constraints import GreaterThan


warnings.filterwarnings("ignore")

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work/")


feature = pd.read_csv('data/feature/feature.csv', index_col=0)
feature_imputation = data_imputation(feature, 'MICE') 
case_gene_update = pd.read_csv('data/processed/variant_clean.csv', index_col=0)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('pathogenic', 1, inplace=True)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('less_pathogenic', 0, inplace=True)
label = case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].reset_index()
label = label['\\12_Candidate variants\\03 Interpretation\\']

#Generating synthetic data based on raw data with/without imputation respectively
real_data_raw = pd.concat([feature, label], axis=1)
real_data_impu = pd.concat([feature_imputation, label], axis=1)
real_data_raw = real_data_raw.rename(columns={"\\12_Candidate variants\\03 Interpretation\\": "label"})
real_data_impu = real_data_impu.rename(columns={"\\12_Candidate variants\\03 Interpretation\\": "label"})

#splitting for imputation real data
feature_real_impu = real_data_impu[real_data_impu.columns[0:-1]]
label_real_impu = real_data_impu[real_data_impu.columns[-1]]
real_data_impu_zero = real_data_impu.loc[real_data_impu[real_data_impu.columns[-1]] == 0]
real_data_impu_one = real_data_impu.loc[real_data_impu[real_data_impu.columns[-1]] == 1]

#splitting for raw real data
feature_real_raw = real_data_raw[real_data_raw.columns[0:-1]]
label_real_raw = real_data_raw[real_data_raw.columns[-1]]
real_data_raw_zero = real_data_raw.loc[real_data_raw[real_data_raw.columns[-1]] == 0]
real_data_raw_one = real_data_raw.loc[real_data_raw[real_data_raw.columns[-1]] == 1]


#############################################################################################################################
#ctgan based on sdv
range_min = pd.DataFrame(index=range(0,500), columns=['range_min'])
range_min = range_min.fillna(0)
range_max = pd.DataFrame(index=range(0,500), columns=['range_max'])
range_max = range_max.fillna(1)
real_data_raw = pd.concat([real_data_raw, range_min.iloc[0:474], range_max.iloc[0:474]], axis=1)
real_data_raw_zero = pd.concat([real_data_raw_zero.reset_index(), range_min.iloc[0:252], range_max.iloc[0:252]], axis=1)
real_data_raw_zero.drop(['index'], axis=1, inplace=True)
real_data_raw_one = pd.concat([real_data_raw_one.reset_index(), range_min.iloc[0:222], range_max.iloc[0:222]], axis=1)
real_data_raw_one.drop(['index'], axis=1, inplace=True)



field_transformers = {'evolutionary age': 'float',
                      'dN/dS': 'float',
                      'gene essentiality': 'one_hot_encoding', 
                      'number of chem interaction action': 'one_hot_encoding',
                      'number of chem interaction': 'one_hot_encoding',
                      'number of chem': 'one_hot_encoding',
                      'number of pathway': 'one_hot_encoding',
                      'number of phenotype': 'one_hot_encoding',
                      'number of rare diseases': 'one_hot_encoding',
                      'number of total diseases': 'one_hot_encoding',
                      'phylogenetic number': 'one_hot_encoding',
                      'net charge value diff': 'one_hot_encoding',
                      'secondary structure value diff': 'one_hot_encoding',
                      'number of hydrogen bond value diff': 'one_hot_encoding',
                      'number of vertices value diff': 'one_hot_encoding',
                      'number of edges value diff': 'one_hot_encoding',
                      'diameter value diff': 'one_hot_encoding'}

#constraints settings for GAN
rare_total_disease_constraint = GreaterThan(
    low='number of rare diseases',
    high='number of total diseases',
    handling_strategy='reject_sampling')

evolutionary_age_constraint = GreaterThan(
    low = 'range_max',
    high = 'evolutionary age',
    handling_strategy='reject_sampling')

dnds_constraint = GreaterThan(
    low = 'range_min',
    high = 'dN/dS',
    handling_strategy='reject_sampling')

gene_haplo_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'haploinsufficiency',
    handling_strategy='reject_sampling')

gene_haplo_max_constraint = GreaterThan(
    low = 'haploinsufficiency',
    high = 'range_max',
    handling_strategy='reject_sampling')

fathmm_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'FATHMM',
    handling_strategy='reject_sampling')

fathmm_max_constraint = GreaterThan(
    low = 'FATHMM',
    high = 'range_max',
    handling_strategy='reject_sampling')

vest_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'VEST',
    handling_strategy='reject_sampling')

vest_max_constraint = GreaterThan(
    low = 'VEST',
    high = 'range_max',
    handling_strategy='reject_sampling')

proven_constraint = GreaterThan(
    low = 'PROVEN',
    high = 'range_min',
    handling_strategy='reject_sampling')

sift_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'SIFT',
    handling_strategy='reject_sampling')

sift_max_constraint = GreaterThan(
    low = 'SIFT',
    high = 'range_max',
    handling_strategy='reject_sampling')


constraints = [rare_total_disease_constraint, evolutionary_age_constraint, dnds_constraint, gene_haplo_min_constraint, 
               gene_haplo_max_constraint, fathmm_min_constraint, fathmm_max_constraint, vest_min_constraint,
               vest_max_constraint, proven_constraint, sift_min_constraint, sift_max_constraint]

#build the model    
model = CTGAN(epochs=300, batch_size=100, field_transformers=field_transformers, constraints=constraints)  #field_distributions=field_distributions

# #Mode 1: generate all samples together (not work well)
# #generate all labels data
# model.fit(real_data_raw)
# sample = model.sample(500)
# sample.drop(['range_min', 'range_max'], axis=1, inplace=True)

# feature_syn_raw = sample[sample.columns[0:-1]]
# label_syn_raw = sample[sample.columns[-1]]
# feature_syn_raw = data_imputation(feature_syn_raw, 'MICE') 

# ss = ShuffleSplit(n_splits=3, test_size=0.33, random_state=0)
# for train_index, test_index in ss.split(real_data_raw):
#     train_x = feature_real_impu.iloc[train_index]
#     train_y = label_real_impu.iloc[train_index]
#     test_x = feature_real_impu.iloc[test_index]
#     test_y = label_real_impu.iloc[test_index]
#     feature_combine, label_combine = merge_data(train_x, train_y, feature_syn_raw, label_syn_raw)

#     rf_baseline(feature_combine, label_combine, test_x, test_y)
#     #xgb_baseline(feature_syn_raw, label_syn_raw, test_x, test_y)



#Mode 2: negative and positive resampling, respectievly
#generate label '0' data of 50000 cases
real_data_raw_zero.drop(['label'], axis=1, inplace=True)
model.fit(real_data_raw_zero)       #model fitting
sample_zero = model.sample(50000)  #generate samples with label '0'
sample_zero.drop(['range_min', 'range_max'], axis=1, inplace=True) #drop 'range_min' and 'range_max' columns
sample_zero['label'] = 0  #add the labels

#generate label '1' data of 50000 cases
real_data_raw_one.drop(['label'], axis=1, inplace=True)
model.fit(real_data_raw_one)
sample_one = model.sample(50000)
sample_one.drop(['range_min', 'range_max'], axis=1, inplace=True)
sample_one['label'] = 1

#concatenate positive and negative synthetic samples
sample_all = pd.concat([sample_zero, sample_one], axis=0)
#sample_all.to_csv('data/synthetic/syn_data_raw.csv')

#remove samples with 'NA' in any of the columns
sample_syn = sample_all.dropna(axis=0,how='any')
#sample_syn.to_csv('data/synthetic/syn_test_raw.csv')


#select 500 synthetic test samples that keeps the similar size of raw data
syn_test_raw = pd.read_csv('data/synthetic/syn_test_raw.csv', index_col=0)
syn_test_raw = syn_test_raw.sample(frac=1)

flag0 = 0
flag1= 0
count_zero = 0
count_one = 0
syn_test_data = []
for i in range(0, len(syn_test_raw)):
    if syn_test_raw['label'].iloc[i] == int(0):
        if count_zero == 250:
            flag0 = 1
        else:
            count_zero = count_zero + 1
            syn_test_data.append(syn_test_raw.iloc[i])
    elif syn_test_raw['label'].iloc[i] == int(1):
        if count_one == 250:
            flag1 = 1
        else:
            count_one = count_one + 1
            syn_test_data.append(syn_test_raw.iloc[i])
    if flag0 == 1 and flag1 == 1:
        break;
syn_test_data = pd.DataFrame(syn_test_data)
syn_test_data['label'] = syn_test_data['label'].astype(int)
syn_test_data.reset_index(inplace=True)
syn_test_data = syn_test_data[syn_test_data.columns[1:40]]

#export synthetic data for external evaluation
syn_test_data.to_csv('data/synthetic/syn_test.csv')     











