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
from sdv.tabular import CopulaGAN

from model import multi_experiment
from model import merge_data
from model import lr_baseline
from model import rf_baseline
from model import xgb_baseline
from model import randomforest_cv
from model import draw_roc_syn_test
from model import draw_roc_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

warnings.filterwarnings("ignore")

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work/")
sys.path.append(os.path.abspath("/Users/rayin/Google Drive/Harvard/5_data/UDN/work/code/"))


feature = pd.read_csv('data/feature/feature.csv', index_col=0)
feature_imputation = data_imputation(feature, 'MICE') 
case_gene_update = pd.read_csv('data/processed/variant_clean.csv', index_col=0)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('pathogenic', 1, inplace=True)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('less_pathogenic', 0, inplace=True)
label = case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].reset_index()
label = label['\\12_Candidate variants\\03 Interpretation\\']
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

field_transformers = {#'evolutionary_age': 'float',
                      #'gene_dnds': 'float',
                      'gene_essentiality': 'one_hot_encoding', 
                      'number_of_chem_interaction_action': 'one_hot_encoding',
                      'number_of_chem_interaction': 'one_hot_encoding',
                      'number_of_chem': 'one_hot_encoding',
                      'number_of_pathway': 'one_hot_encoding',
                      'number_of_phenotype': 'one_hot_encoding',
                      'number_of_rare_disease': 'one_hot_encoding',
                      'number_of_total_disease': 'one_hot_encoding',
                      'phylogenetic_number': 'one_hot_encoding',
                      'net_charge_value_diff': 'one_hot_encoding',
                      'secondary_structure_value_diff': 'one_hot_encoding',
                      'number_of_hydrogen_bond_value_diff': 'one_hot_encoding',
                      'number_of_vertices_value_diff': 'one_hot_encoding',
                      'number_of_edges_value_diff': 'one_hot_encoding',
                      'diameter_value_diff': 'one_hot_encoding'}

#constraints settings for GAN
rare_total_disease_constraint = GreaterThan(
    low='number_of_rare_disease',
    high='number_of_total_disease',
    handling_strategy='reject_sampling')

evolutionary_age_constraint = GreaterThan(
    low = 'range_max',
    high = 'evolutionary_age',
    handling_strategy='reject_sampling')

dnds_constraint = GreaterThan(
    low = 'range_min',
    high = 'gene_dnds',
    handling_strategy='reject_sampling')

gene_haplo_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'gene_haplo',
    handling_strategy='reject_sampling')

gene_haplo_max_constraint = GreaterThan(
    low = 'gene_haplo',
    high = 'range_max',
    handling_strategy='reject_sampling')

fathmm_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'fathmm',
    handling_strategy='reject_sampling')

fathmm_max_constraint = GreaterThan(
    low = 'fathmm',
    high = 'range_max',
    handling_strategy='reject_sampling')

vest_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'vest',
    handling_strategy='reject_sampling')

vest_max_constraint = GreaterThan(
    low = 'vest',
    high = 'range_max',
    handling_strategy='reject_sampling')

proven_constraint = GreaterThan(
    low = 'proven',
    high = 'range_min',
    handling_strategy='reject_sampling')

sift_min_constraint = GreaterThan(
    low = 'range_min',
    high = 'sift',
    handling_strategy='reject_sampling')

sift_max_constraint = GreaterThan(
    low = 'sift',
    high = 'range_max',
    handling_strategy='reject_sampling')

# field_distributions = {
#     'evolutionary_age': 'gamma',
#     'gene_dnds': 'gamma'}

constraints = [rare_total_disease_constraint, evolutionary_age_constraint, dnds_constraint, gene_haplo_min_constraint, 
               gene_haplo_max_constraint, fathmm_min_constraint, fathmm_max_constraint, vest_min_constraint,
               vest_max_constraint, proven_constraint, sift_min_constraint, sift_max_constraint]
    
model = CTGAN(epochs=300, batch_size=100, field_transformers=field_transformers, constraints=constraints)  #field_distributions=field_distributions

#model 1: generate all samples together (not work well)
#generate all labels data
model.fit(real_data_raw)
sample = model.sample(500)
sample.drop(['range_min', 'range_max'], axis=1, inplace=True)

feature_syn_raw = sample[sample.columns[0:-1]]
label_syn_raw = sample[sample.columns[-1]]
feature_syn_raw = data_imputation(feature_syn_raw, 'MICE') 

ss = ShuffleSplit(n_splits=3, test_size=0.33, random_state=0)
for train_index, test_index in ss.split(real_data_raw):
    train_x = feature_real_impu.iloc[train_index]
    train_y = label_real_impu.iloc[train_index]
    test_x = feature_real_impu.iloc[test_index]
    test_y = label_real_impu.iloc[test_index]
    feature_combine, label_combine = merge_data(train_x, train_y, feature_syn_raw, label_syn_raw)

    rf_baseline(feature_combine, label_combine, test_x, test_y)
    #xgb_baseline(feature_syn_raw, label_syn_raw, test_x, test_y)



#Mode 2: negative and positive resampling
#generate label '0' data
real_data_raw_zero.drop(['label'], axis=1, inplace=True)
model.fit(real_data_raw_zero)
sample_zero = model.sample(50000)
sample_zero.drop(['range_min', 'range_max'], axis=1, inplace=True)
sample_zero['label'] = 0

#generate label '1' data
real_data_raw_one.drop(['label'], axis=1, inplace=True)
model.fit(real_data_raw_one)
sample_one = model.sample(50000)
sample_one.drop(['range_min', 'range_max'], axis=1, inplace=True)
sample_one['label'] = 1


sample_all = pd.concat([sample_zero, sample_one], axis=0)
#sample_all.to_csv('data/synthetic/syn_data_raw.csv')
sample_syn = sample_all.dropna(axis=0,how='any')
#sample_syn.to_csv('data/synthetic/syn_test_raw.csv')

# #generate synthetic test data -- scenario 1
# syn_test_sample = sample_syn.sample(n=475, axis=0, replace=False)
# syn_test_sample.reset_index(inplace=True)
# syn_test_sample = syn_test_sample[syn_test_sample.columns[1:40]]
# syn_test_sample.to_csv('data/synthetic/syn_test.csv')

#generate synthetic test data scenario 2
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
syn_test_data.to_csv('data/synthetic/syn_test.csv')     

train_x, test_x, train_y, test_y = train_test_split(feature_imputation, label, test_size=0.2)

# test_1 = syn_test_sample.sample(n=95, axis=0, replace=False)
# feature_syn_raw = test_1[test_1.columns[0:-1]]
# label_syn_raw = test_1[test_1.columns[-1]]

# rf_baseline(train_x, train_y, test_x, test_y)
# rf_baseline(train_x, train_y, feature_syn_raw, label_syn_raw)
# rf_baseline(pd.concat([train_x, test_x], axis=0), pd.concat([train_y, test_y], axis=0), feature_syn_raw, label_syn_raw)

#multi_experiment('rf', 1, feature_imputation, label)



def subset(alist, idxs):
    '''
        fetch the subset of alist based on idxs
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list


def split_list(alist, group_num=5, shuffle=True, retain_left=False):
    '''
        alist divide into sublist by group that for each subgroup we have len(alist)//group elements
        shuffle: if random, True by default
        retain_left: the remaining elements will be taken as one group
    '''

    index = list(range(len(alist))) 

    if shuffle: 
        random.shuffle(index) 
    
    elem_num = len(alist) // group_num 
    sub_lists = {}
    
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists['set'+str(idx)] = subset(alist, index[start:end])
    
    if retain_left and group_num * elem_num != len(index): 
        sub_lists['set'+str(idx+1)] = subset(alist, index[end:])
    
    return sub_lists




syn_test_index = syn_test_sample.index
syn_data_split = split_list(syn_test_index, group_num=5, shuffle=True, retain_left=False)
    
for key, value in syn_data_split.items():
    #print(key)
    syn_test = syn_test_sample.iloc[value]
    syn_test_x = syn_test[syn_test.columns[0:-1]]
    syn_test_y = syn_test[syn_test.columns[-1]]
    rf_baseline(train_x, train_y, syn_test_x, syn_test_y)




feature_combine, label_combine = merge_data(feature_real_raw, label_real_raw, feature_syn_raw, label_syn_raw)
#feature_combine, label_combine = merge_data(feature_real_impu, label_real_impu, feature_syn_raw, label_syn_raw)
feature_combine = data_imputation(feature_combine, 'MICE') 
sample_all = data_imputation(sample_all, 'MICE')
feature_syn_raw = feature_combine.iloc[474:2474]
label_syn_raw = label_combine.iloc[474:2474]
feature_real_impu = feature_combine.iloc[0:474]
label_real_impu = label_combine.iloc[0:474]

ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_index, test_index in ss.split(real_data_raw):
    train_x = feature_real_impu.iloc[train_index]
    train_y = label_real_impu.iloc[train_index]
    test_x = feature_real_impu.iloc[test_index]
    test_y = label_real_impu.iloc[test_index]
    #feature_combine, label_combine = merge_data(train_x, train_y, feature_syn_raw, label_syn_raw)
    #lr_baseline(feature_combine, label_combine, test_x, test_y)
    
    #rf_baseline(train_x, train_y, test_x, test_y)
    #rf_baseline(feature_combine, label_combine, test_x, test_y)
    
    #xgb_baseline(train_x, train_y, test_x, test_y)
    xgb_baseline(feature_combine, label_combine, test_x, test_y)

test_sample = sample_all.sample(10)
test_syn_x = test_sample.drop(['label'], axis=1)
test_syn_y = test_sample['label']

xgb_baseline(feature_real_impu, label_real_impu, test_syn_x, test_syn_y)
rf_baseline(feature_real_impu, label_real_impu, test_syn_x, test_syn_y)

# #model.save('my_model.pkl')
# #loaded = CTGAN.load('my_model.pkl')

# #evaluaton for synthetic data using CTGAN
# evaluate(synthetic_data, feature)
# LogisticDetection.compute(synthetic_data, feature)
# SVCDetection.compute(synthetic_data, feature)

# CSTest.compute(feature, synthetic_data)
# KSTest.compute(feature, synthetic_data)


  
        







#########################################################################################################################
#ctgan method
#model training
discrete_columns = ['gene_essentiality', 'number_of_chem_interaction_action', 'number_of_chem_interaction', 'number_of_chem',
                    'number_of_pathway', 'number_of_phenotype', 'number_of_rare_disease', 'number_of_total_disease',
                    'phylogenetic_number', 'number_of_vertices_value_diff', 'number_of_edges_value_diff',
                    'label']
ctgan = CTGANSynthesizer(epochs=300)

#Mode 1: generate all samples without labeling
ctgan.fit(real_data_impu, discrete_columns)
samples = ctgan.sample(500)

feature_syn_impu = samples[samples.columns[0:-1]]
label_syn_impu = samples[samples.columns[-1]]


ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in ss.split(real_data_impu):
    train_x = feature_real_impu.iloc[train_index]
    train_y = label_real_impu.iloc[train_index]
    test_x = feature_real_impu.iloc[test_index]
    test_y = label_real_impu.iloc[test_index]
    feature_combine, label_combine = merge_data(train_x, train_y, feature_syn_impu, label_syn_impu)
    #lr_baseline(feature_combine, label_combine, test_x, test_y)
    #rf_baseline(feature_syn_impu, label_syn_impu, test_x, test_y)
    #xgb_baseline(feature_combine, label_combine, test_x, test_y)
    xgb_baseline(feature_syn_impu, label_syn_impu, test_x, test_y)



#########################################################################################################################
#Mode 2: negative and positive resampling
ctgan.fit(real_data_impu_zero, discrete_columns)
sample_zero = ctgan.sample(250)

ctgan.fit(real_data_impu_one, discrete_columns)
sample_one = ctgan.sample(250)

sample_all = pd.concat([sample_zero, sample_one], axis=0)
feature_syn_impu = sample_all[sample_all.columns[0:-1]]
label_syn_impu = sample_all[sample_all.columns[-1]]

feature_combine, label_combine = merge_data(feature_real_impu, label_real_impu, feature_syn_impu, label_syn_impu)

#Mode 3: negative and positive resampling with only real data for testing
# kf = KFold(n_splits=3)
# for train, test in kf.split(real_data_impu):
#     print("%s %s" % (train, test))

ss = ShuffleSplit(n_splits=3, test_size=0.33, random_state=0)
for train_index, test_index in ss.split(real_data_impu):
    train_x = feature_real_impu.iloc[train_index]
    train_y = label_real_impu.iloc[train_index]
    test_x = feature_real_impu.iloc[test_index]
    test_y = label_real_impu.iloc[test_index]
    feature_combine, label_combine = merge_data(train_x, train_y, feature_syn_impu, label_syn_impu)
    #lr_baseline(feature_combine, label_combine, test_x, test_y)
    #rf_baseline(feature_syn_impu, label_syn_impu, test_x, test_y)
    xgb_baseline(feature_syn_impu, label_syn_impu, test_x, test_y)


    #print("%s %s" % (train_index, test_index))
    
train_x, test_x, train_y, test_y = train_test_split(feature_real_impu, label_real_impu, test_size=0.2)
feature_combine, label_combine = merge_data(train_x, train_y, feature_syn_impu, label_syn_impu) #only for training


#training model
lr_baseline(feature_combine, label_combine, test_x, test_y)
rf_baseline(feature_combine, label_combine, test_x, test_y)
xgb_baseline(feature_combine, label_combine, test_x, test_y)




multi_experiment('xgboost', 1, feature_syn_impu, label_syn_impu)
multi_experiment('rf', 1, feature_combine, label_combine)




















