#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:35:52 2021

@author: rayin
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import warnings
import torchvision.models as models
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

sys.path.append(os.path.abspath("/Users/rayin/Google Drive/Harvard/5_data/UDN/work/code/"))

from model import subset
from model import split_list

warnings.filterwarnings("ignore")

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

feature = pd.read_csv('data/feature/feature.csv', index_col=0)
case_gene_update = pd.read_csv('data/processed/variant_clean.csv', index_col=0)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('pathogenic', 1, inplace=True)
case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].replace('less_pathogenic', 0, inplace=True)
label = case_gene_update['\\12_Candidate variants\\03 Interpretation\\'].reset_index()
label = label[label.columns[1]]

sift_score = feature['SIFT']
proven_score = feature['PROVEN']
fathmm_score = feature['FATHMM']
cadd_score = feature['CADD']
vest_score = feature['VEST']
polyphen2_humdiv_score = pd.read_csv('data/score/polyphen2_humdiv.csv', index_col=0)
polyphen2_humvar_score = pd.read_csv('data/score/polyphen2_humvar.csv', index_col=0)

#mode contains positive and negative types. Positive: fathmm, vest, cadd, polyphen2   Negative: sift, proven
def score_to_label(score_value, threshold, mode):
    label = []
    if mode == 'negative':
        for i in range(0, len(score_value)):
            if np.isnan(score_value[i]) == True:
                label.append(score_value[i])
            elif score_value[i] >= threshold:
                label.append(0)
            elif score_value[i] < threshold:
                label.append(1)
    elif mode == 'positive':
        for i in range(0, len(score_value)):
            if np.isnan(score_value[i]) == True:
                label.append(score_value[i])
            elif score_value[i] >= threshold:
                label.append(1)
            elif score_value[i] < threshold:
                label.append(0)
    label = pd.Series(label)
    return label

def compare_prediction(label, compare_label):
    label_index = label.index
    label_split = split_list(label_index, group_num=10, shuffle=False, retain_left=False)
    
    count = []
    acc = []
    pre = []
    rec = []
    f1 = []
    mcc = []
    auc = []
    for key, value in label_split.items():
        true_label = []
        prediction_label_subset = []
        count_nan = 0
    
        for i in range(0, len(value)):
            if np.isnan(compare_label[value[i]]) == True:
                count_nan = count_nan + 1
            else:
                true_label.append(label[value[i]])
                prediction_label_subset.append(int(compare_label[value[i]]))
    
        count.append(count_nan)
        acc.append(accuracy_score(true_label, prediction_label_subset))
        pre.append(precision_score(true_label, prediction_label_subset))
        rec.append(recall_score(true_label, prediction_label_subset))    
        f1.append(f1_score(true_label, prediction_label_subset))
        mcc.append(matthews_corrcoef(true_label, prediction_label_subset))
        auc.append(roc_auc_score(true_label, prediction_label_subset))
    
    count_mean = np.mean(count_nan)
    count_sd = np.round(np.std(count, ddof = 1), 3)
    acc_mean = np.round(np.mean(acc), 3)
    acc_sd = np.round(np.std(acc, ddof = 1), 3)
    pre_mean = np.round(np.mean(pre), 3)
    pre_sd = np.round(np.std(pre, ddof = 1), 3)
    rec_mean = np.round(np.mean(rec), 3)
    rec_sd = np.round(np.std(rec, ddof = 1), 3)
    f1_mean = np.round(np.mean(f1), 3)
    f1_sd = np.round(np.std(f1, ddof = 1), 3)
    mcc_mean = np.round(np.mean(mcc), 3)
    mcc_sd = np.round(np.std(mcc, ddof = 1), 3)
    auc_mean = np.round(np.mean(auc), 3)
    auc_sd = np.round(np.std(auc, ddof = 1), 3)
    
    print('NA:%d' %count_mean + ' ' + "sd:%f"%count_sd + '\n')
    print("accuracy:%f" %acc_mean + ' ' + "sd:%f"%acc_sd + '\n')
    print("precision:%f" %pre_mean + ' ' + "sd:%f"%pre_sd + '\n')        
    print("recall:%f" %rec_mean + ' ' + "sd:%f"%rec_sd + '\n')       
    print("f1:%f" %f1_mean + ' ' + "sd:%f"%f1_sd + '\n')
    print("mcc:%f" %mcc_mean + ' ' + "sd:%f"%mcc_sd + '\n')
    print("auc:%f" %auc_mean + ' ' + "sd:%f"%auc_sd + '\n')
    
    output = [acc_mean, acc_sd, pre_mean, pre_sd, rec_mean, rec_sd, f1_mean, f1_sd, mcc_mean, mcc_sd, auc_mean, auc_sd, count_mean, count_sd]
    output = pd.DataFrame(output)
    output = output.T
    return output


sift_label = score_to_label(sift_score, 0.05, 'negative')
polyphen2_humdiv_label = score_to_label(polyphen2_humdiv_score, 0.5, 'positive')
polyphen2_humvar_label = score_to_label(polyphen2_humvar_score, 0.5, 'positive')
proven_label = score_to_label(proven_score, -2.5, 'negative')
fathmm_label = score_to_label(fathmm_score, 0.5, 'positive')
vest_label = score_to_label(vest_score, 0.5, 'positive')
cadd_label = score_to_label(cadd_score, 1.9, 'positive')


#https://sift.bii.a-star.edu.sg/
#http://genetics.bwh.harvard.edu/pph2/bgi.shtml
#http://provean.jcvi.org/index.php
#http://fathmm.biocompute.org.uk/
#http://cravat.us/CRAVAT/
#https://cadd.gs.washington.edu/

sift_result = compare_prediction(label, sift_label)
polyphen2_humdiv_result = compare_prediction(label, polyphen2_humdiv_label)
polyphen2_humvar_result = compare_prediction(label, polyphen2_humvar_label)
proven_result = compare_prediction(label, proven_label)
fathmm_result = compare_prediction(label, fathmm_label)
vest_result = compare_prediction(label, vest_label)
cadd_result = compare_prediction(label, cadd_label)   
   

#mutation_taster prediction results
mutation_taster = pd.read_excel('data/score/mutation_taster.xlsx')
mutation_taster = mutation_taster[mutation_taster['genesymbol'] != 'compound']
mutation_taster = mutation_taster.drop_duplicates(subset = ['chromosome', 'position', 'genesymbol'], keep = 'first')

# def label_transform(score):
#     for i in range(0, len(score)):
#         if score['prediction'].iloc[i] == 'disease_causing_automatic':
#             if score['probability'].iloc[i] >= 0.5:
#                 score['prediction'].iloc[i] = 1
#             else:
#                 score['prediction'].iloc[i] = 0
#         elif score['prediction'].iloc[i] == 'disease_causing':
#             if score['probability'].iloc[i] >= 0.5:
#                 score['prediction'].iloc[i] = 1
#             else:
#                 score['prediction'].iloc[i] = 0
#         elif score['prediction'].iloc[i] == 'polymorphism_automatic':
#             if score['probability'].iloc[i] >= 0.5:
#                 score['prediction'].iloc[i] = 0
#             else:
#                 score['prediction'].iloc[i] = 1
#         elif score['prediction'].iloc[i] == 'polymorphism':
#             if score['probability'].iloc[i] >= 0.5:
#                 score['prediction'].iloc[i] = 0
#             else:
#                 score['prediction'].iloc[i] = 1
#     return score

# mutation_taster = label_transform(mutation_taster)


mutation_taster.prediction[mutation_taster['prediction'] == 'disease_causing_automatic'] = 1
mutation_taster.prediction[mutation_taster['prediction'] == 'disease_causing'] = 1
mutation_taster.prediction[mutation_taster['prediction'] == 'polymorphism_automatic'] = 0
mutation_taster.prediction[mutation_taster['prediction'] == 'polymorphism'] = 0

#mutation taster label
mutation_taster_label = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(mutation_taster)):
        if case_gene_update['START'].iloc[i] == mutation_taster['position'].iloc[j]:
            if case_gene_update['\\11_Candidate genes\\Gene Name\\'].iloc[i] == mutation_taster['genesymbol'].iloc[j]:
                flag = 1
                break;
    if flag == 1:
        mutation_taster_label.append(mutation_taster['prediction'].iloc[j])
    elif flag == 0:
        mutation_taster_label.append(float('nan'))

mutation_taster_label = pd.Series(mutation_taster_label) 
mu_taster_result = compare_prediction(label, mutation_taster_label)


##################################################################################################################################

#mutation assessor prediction results
mutation_assessor = pd.read_excel('data/score/mutation_assessor.xlsx')
#mutation assessor label
mutation_assessor_label = []
for i in range(0, len(mutation_assessor)):
    if str(mutation_assessor['impact'].iloc[i]) == str('nan'):
        mutation_assessor_label.append(float('nan'))
    elif mutation_assessor['impact'].iloc[i] == 'high':
        mutation_assessor_label.append(1)
    elif mutation_assessor['impact'].iloc[i] == 'medium':
        mutation_assessor_label.append(1)
    elif mutation_assessor['impact'].iloc[i] == 'neutral':
        mutation_assessor_label.append(0)
    elif mutation_assessor['impact'].iloc[i] == 'low':
        mutation_assessor_label.append(0)

mutation_assessor_label = pd.Series(mutation_assessor_label)
mu_assessor_result = compare_prediction(label, mutation_assessor_label)


comparison_result = pd.concat([sift_result, polyphen2_humdiv_result, polyphen2_humvar_result, proven_result, fathmm_result, 
                               vest_result, mu_taster_result ,mu_assessor_result, cadd_result], axis = 0)








