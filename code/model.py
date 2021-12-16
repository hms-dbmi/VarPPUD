#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:48:48 2021

@author: rayin
"""

import os, sys
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch
from collections import Counter
from scipy import interp
from torch import nn

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

def is_input_contain_nan(df):
    # for i in range(0, df.shape[0]):
    #     for j in range(0, df.shape[1]):    
    #         if str(df.iloc[i][j]).lower() == 'nan':
    #             print('row'%i + 'contains nan at'%j + '\n')
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return indices_to_keep

                
def logistic_regression_cv(data_x, data_y):
    
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    kfold = StratifiedKFold(n_splits=5) #shuffle=True
    
    #training
    train_acc = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='accuracy')
    train_pre = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='precision')
    train_rec = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='recall')
    train_f1 = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='f1')
    train_pred = cross_val_predict(clf, train_x, train_y, cv=kfold)
    train_mcc = matthews_corrcoef(train_y, train_pred)
    
    #testing
    clf = LogisticRegression().fit(train_x, train_y)
    test_acc = accuracy_score(test_y, clf.predict(test_x))
    test_pre = precision_score(test_y, clf.predict(test_x))
    test_rec = recall_score(test_y, clf.predict(test_x))
    test_f1 = f1_score(test_y, clf.predict(test_x))
    test_mcc = matthews_corrcoef(test_y, clf.predict(test_x))
    
    return train_acc, train_pre, train_rec, train_f1, train_mcc, test_acc, test_pre, test_rec, test_f1, test_mcc


def randomforest_cv(data_x, data_y):
    #np.random.seed(100)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, random_state=42)
    #print(test_y.index.values)
    #cross-validation
    #clf = ensemble.RandomForestClassifier(n_estimators=140) 
    clf = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                  min_samples_leaf=10, random_state=42)
    kfold = StratifiedKFold(n_splits=5) #shuffle=True
    #kfold = KFold(n_splits=5, shuffle=True)
    train_acc = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='accuracy')
    train_pre = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='precision')
    train_rec = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='recall')
    train_f1 = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='f1')
    train_pred = cross_val_predict(clf, train_x, train_y, cv=kfold)
    train_mcc = matthews_corrcoef(train_y, train_pred)
    train_auc = roc_auc_score(train_y, train_pred)
    
    # print('training set:')
    # print("train_acc: %f" %train_acc.mean() + '\n')
    # print("train_pre: %f" %train_pre.mean() + '\n')
    # print("train_rec: %f" %train_rec.mean() + '\n')
    # print("train_f1 : %f " %train_f1.mean() + '\n')
    # print("train_mcc: %f" %np.mean(train_mcc) + '\n')
    
    #testing
    clf = ensemble.RandomForestClassifier().fit(train_x, train_y)
    test_acc = accuracy_score(test_y, clf.predict(test_x))
    test_pre = precision_score(test_y, clf.predict(test_x))
    test_rec = recall_score(test_y, clf.predict(test_x))
    test_f1 = f1_score(test_y, clf.predict(test_x))
    test_mcc = matthews_corrcoef(test_y, clf.predict(test_x))
    test_auc = roc_auc_score(test_y, clf.predict(test_x))
    
    # print('testing set:')
    # print("test_acc: %f" %test_acc.mean() + '\n')
    # print("test_pre: %f" %test_pre.mean() + '\n')
    # print("test_rec: %f" %test_rec.mean() + '\n')
    # print("test_f1 : %f " %test_f1.mean() + '\n')
    # print("test_mcc: %f" %np.mean(test_mcc) + '\n')
    
    return train_acc, train_pre, train_rec, train_f1, train_mcc, train_auc, test_acc, test_pre, test_rec, test_f1, test_mcc, test_auc

#randomforest_cv(feature, label)



def xgboost_cv(data_x, data_y):
    np.random.seed(100)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, random_state=42)
    #cross-validation
    clf = xgb.XGBClassifier(max_depth=8,
                        learning_rate=0.05,
                        n_estimators=20,
                        objective='binary:logistic',
                        eval_metric = "logloss",
                        nthread=4,
                        gamma=0,
                        min_child_weight=5,
                        max_delta_step=0,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=1e-5,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=27,
                        missing=None)
    
    #clf = xgb.XGBClassifier(n_estimators=30)
    #kfold = KFold(n_splits=10, shuffle=True)
    kfold = StratifiedKFold(n_splits=5) 
    
    #training
    train_acc = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='accuracy')
    train_pre = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='precision')
    train_rec = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='recall')
    train_f1 = cross_val_score(clf, train_x, train_y, cv=kfold, scoring='f1')
    train_pred = cross_val_predict(clf, train_x, train_y, cv=kfold)
    train_mcc = matthews_corrcoef(train_y, train_pred)
    
    #testing
    clf = xgb.XGBClassifier().fit(train_x, train_y)
    test_acc = accuracy_score(test_y, clf.predict(test_x))
    test_pre = precision_score(test_y, clf.predict(test_x))
    test_rec = recall_score(test_y, clf.predict(test_x))
    test_f1 = f1_score(test_y, clf.predict(test_x))
    test_mcc = matthews_corrcoef(test_y, clf.predict(test_x))
    
    # print('training set:')
    # print("train_acc: %f" %train_acc.mean() + '\n')
    # print("train_pre: %f" %train_pre.mean() + '\n')
    # print("train_rec: %f" %train_rec.mean() + '\n')
    # print("train_f1 : %f " %train_f1.mean() + '\n')
    # print("train_mcc: %f" %np.mean(train_mcc) + '\n')
    
    # print('testing set:')
    # print("test_acc: %f" %test_acc.mean() + '\n')
    # print("test_pre: %f" %test_pre.mean() + '\n')
    # print("test_rec: %f" %test_rec.mean() + '\n')
    # print("test_f1 : %f " %test_f1.mean() + '\n')
    # print("test_mcc: %f" %np.mean(test_mcc) + '\n')
    
    return train_acc, train_pre, train_rec, train_f1, train_mcc, test_acc, test_pre, test_rec, test_f1, test_mcc
#xgboost_cv(feature_imputation, label)       



def patient_variable_influence(model, data_x, data_y, patient_info, variable):    
    #cross-validation on raw features + patient_level variables (after one-hot coding)
    if variable == 'raw':
        feature_new = data_x
    elif variable == 'age_symptom_onset':
        patient_variable = patient_info.iloc[:, 0:1]
        feature_new = pd.concat([data_x, patient_variable], axis=1)
    elif variable == 'current_age':
        patient_variable = patient_info.iloc[:, 1:2]
        feature_new = pd.concat([data_x, patient_variable], axis=1)
    elif variable == 'ethnicity':
        patient_variable = patient_info.iloc[:, 2:6]
        feature_new = pd.concat([data_x, patient_variable], axis=1)
    elif variable == 'gender':
        patient_variable = patient_info.iloc[:, 6:8]
        feature_new = pd.concat([data_x, patient_variable], axis=1)
    elif variable == 'race':
        patient_variable = patient_info.iloc[:, 8:14]
        feature_new = pd.concat([data_x, patient_variable], axis=1)
    elif variable == 'all':  #all features
        feature_new = pd.concat([data_x, patient_info], axis=1)
        
    #np.random.seed(100)

    if model == 'rf':
        train_acc, train_pre, train_rec, train_f1, train_mcc, train_auc, test_acc, test_pre, test_rec, test_f1, test_mcc, test_auc = randomforest_cv(feature_new, data_y)

    elif model == 'xgboost':
        train_acc, train_pre, train_rec, train_f1, train_mcc, train_auc, test_acc, test_pre, test_rec, test_f1, test_mcc, test_auc = xgboost_cv(feature_new, data_y)
    
    print(model + ' + ' + variable + ':')
    #print("test_acc: %f" %test_acc.mean() + '\n')
    #print("test_pre: %f" %test_pre.mean() + '\n')
    #print("test_rec: %f" %test_rec.mean() + '\n')
    #print("test_f1 : %f " %test_f1.mean() + '\n')
    #print("test_mcc: %f" %np.mean(test_mcc) + '\n')
    print("test_auc : %f " %test_auc.mean() + '\n')
    
    
def multi_experiment(model, time, train_x, train_y):
    train_acc_all = []
    train_pre_all = []
    train_rec_all = []
    train_f1_all = []
    train_mcc_all = []
    train_auc_all = []
    
    test_acc_all = []
    test_pre_all = []
    test_rec_all = []
    test_f1_all = []
    test_mcc_all = []
    test_auc_all = []
    
    
    for i in range(0, time):
        if model == 'rf':
            if sum(train_x.isnull().sum().values) != 0:
                imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
                x_new = pd.DataFrame(imp.fit_transform(train_x))
                train_x = x_new
            train_acc, train_pre, train_rec, train_f1, train_mcc, train_auc, test_acc, test_pre, test_rec, test_f1, test_mcc, test_auc = randomforest_cv(train_x, train_y)
        elif model == 'lr':
            if sum(train_x.isnull().sum().values) != 0:
                imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
                x_new = pd.DataFrame(imp.fit_transform(train_x))
                train_x = x_new          
            train_acc, train_pre, train_rec, train_f1, train_mcc, test_acc, test_pre, test_rec, test_f1, test_mcc = logistic_regression_cv(train_x, train_y)
        elif model == 'xgboost':  
            train_acc, train_pre, train_rec, train_f1, train_mcc, test_acc, test_pre, test_rec, test_f1, test_mcc = xgboost_cv(train_x, train_y)
            
        train_acc_all.append(train_acc)
        train_pre_all.append(train_pre)
        train_rec_all.append(train_rec)
        train_f1_all.append(train_f1)
        train_mcc_all.append(train_mcc)
        train_auc_all.append(train_auc)
        
        test_acc_all.append(test_acc)
        test_pre_all.append(test_pre)
        test_rec_all.append(test_rec)
        test_f1_all.append(test_f1)
        test_mcc_all.append(test_mcc)
        test_auc_all.append(test_auc)
        
    train_acc_mean = np.mean(train_acc_all)
    train_acc_sd = np.std(train_acc_all, ddof = 1)
    train_pre_mean = np.mean(train_pre_all)
    train_pre_sd = np.std(train_pre_all, ddof = 1)
    train_rec_mean = np.mean(train_rec_all)
    train_rec_sd = np.std(train_rec_all, ddof = 1)
    train_f1_mean = np.mean(train_f1_all)
    train_f1_sd = np.std(train_f1_all, ddof = 1)
    train_mcc_mean = np.mean(train_mcc_all)
    train_mcc_sd = np.std(train_mcc_all, ddof = 1)
    train_auc_mean = np.mean(train_auc_all)
    train_auc_sd = np.std(train_auc_all, ddof = 1)
    
    test_acc_mean = np.mean(test_acc_all)
    test_acc_sd = np.std(test_acc_all, ddof = 1)
    test_pre_mean = np.mean(test_pre_all)
    test_pre_sd = np.std(test_pre_all, ddof = 1)
    test_rec_mean = np.mean(test_rec_all)
    test_rec_sd = np.std(test_rec_all, ddof = 1)
    test_f1_mean = np.mean(test_f1_all)
    test_f1_sd = np.std(test_f1_all, ddof = 1)
    test_mcc_mean = np.mean(test_mcc_all)
    test_mcc_sd = np.std(test_mcc_all, ddof = 1)
    test_auc_mean = np.mean(test_auc_all)
    test_auc_sd = np.std(test_auc_all, ddof = 1)
    
    print(model + '_train')
    print("accuracy:%f" %train_acc_mean + ' ' + "sd:%f"%train_acc_sd + '\n')
    print("precision:%f" %train_pre_mean + ' ' + "sd:%f"%train_pre_sd + '\n')        
    print("recall:%f" %train_rec_mean + ' ' + "sd:%f"%train_rec_sd + '\n')       
    print("f1:%f" %train_f1_mean + ' ' + "sd:%f"%train_f1_sd + '\n')
    print("auc:%f" %train_auc_mean + ' ' + "sd:%f"%train_auc_sd + '\n')
    #print("mcc:%f" %train_mcc_mean + ' ' + "sd:%f"%train_mcc_sd + '\n')
    
    print(model + '_test') 
    print("accuracy:%f" %test_acc_mean + ' ' + "sd:%f"%test_acc_sd + '\n')
    print("precision:%f" %test_pre_mean + ' ' + "sd:%f"%test_pre_sd + '\n')        
    print("recall:%f" %test_rec_mean + ' ' + "sd:%f"%test_rec_sd + '\n')       
    print("f1:%f" %test_f1_mean + ' ' + "sd:%f"%test_f1_sd + '\n')
    print("auc:%f" %test_auc_mean + ' ' + "sd:%f"%test_auc_sd + '\n')
    #print("mcc:%f" %test_mcc_mean + ' ' + "sd:%f"%test_mcc_sd + '\n')
    

#cross-validation train + independent test data to draw roc curve
def draw_roc_train_test_split(data_x, data_y, syn_data, model):
    np.random.seed(100)
    if model == 'lr':
        if sum(data_x.isnull().sum().values) != 0:
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
            data_x = pd.DataFrame(imp.fit_transform(data_x))
        clf = LogisticRegression()
    elif model == 'rf':
        if sum(data_x.isnull().sum().values) != 0:
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
            data_x = pd.DataFrame(imp.fit_transform(data_x))       
        clf = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                  min_samples_leaf=10, random_state=42)
    elif model == 'xgboost':  
        clf = xgb.XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=20, objective='binary:logistic', nthread=4, gamma=0, min_child_weight=5,
                                max_delta_step=0, subsample=0.8, colsample_bytree=0.7, colsample_bylevel=1, reg_alpha=1e-5, reg_lambda=1,
                                scale_pos_weight=1, seed=27, missing=None)
    else:
        print('error')
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)
    
    #plot the roc curve
    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.plot(fpr, tpr, label= 'Real set'  + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=1, alpha=.8)
    
    #plot the roc curve for synthetic test set
    syn_test_index = syn_data.index
    syn_data_split = split_list(syn_test_index, group_num=10, shuffle=True, retain_left=False)
    
    for key, value in syn_data_split.items():
        #print(key)
        syn_test = syn_data.iloc[value]
        syn_test_x = syn_test[syn_test.columns[0:-1]]
        syn_test_y = syn_test[syn_test.columns[-1]]

        probas_ = clf.fit(train_x, train_y).predict_proba(syn_test_x)
        fpr, tpr, thresholds = roc_curve(syn_test_y, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=.8)
        plt.plot(fpr, tpr, label= 'Syn' + ' ' + key + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=1, alpha=.8)
    
    
    plt.xlim([-0.00, 1.00])
    plt.ylim([-0.00, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    #plt.show()
    
#cross validation to draw roc curve
def draw_roc_curve(train_x, train_y, syn_data, model):
    np.random.seed(100)
    if model == 'lr':
        if sum(train_x.isnull().sum().values) != 0:
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
            train_x = pd.DataFrame(imp.fit_transform(train_x))
        clf = LogisticRegression()
    elif model == 'rf':
        if sum(train_x.isnull().sum().values) != 0:
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
            train_x = pd.DataFrame(imp.fit_transform(train_x))       
        clf = ensemble.RandomForestClassifier()
    elif model == 'xgboost':  
        clf = xgb.XGBClassifier()
    else:
        print('error')

    #plot the roc curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    for train, test in kfold.split(train_x, train_y):
        probas_ = clf.fit(train_x.iloc[train], train_y.iloc[train]).predict_proba(train_x.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(train_y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i = i + 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, label= 'Real set' + ' ' + r'(AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=1, alpha=.8)
    
    #std_tpr = np.std(tprs, axis=0)
    #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    #plot the roc curve for synthetic test set
    syn_test_index = syn_data.index
    syn_data_split = split_list(syn_test_index, group_num=5, shuffle=True, retain_left=True)
    
    for key, value in syn_data_split.items():
        #print(key)
        syn_test = syn_data.iloc[value]
        syn_test_x = syn_test[syn_test.columns[0:-1]]
        syn_test_y = syn_test[syn_test.columns[-1]]

        probas_ = clf.fit(train_x, train_y).predict_proba(syn_test_x)
        fpr, tpr, thresholds = roc_curve(syn_test_y, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=.8)
        plt.plot(fpr, tpr, label= 'Syn' + ' ' + key + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=1, alpha=.8)
    
    plt.xlim([-0.00, 1.00])
    plt.ylim([-0.00, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    #plt.show()

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

    if shuffle == True: 
        random.shuffle(index) 
    
    elem_num = len(alist) // group_num 
    sub_lists = {}
    
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists['set'+str(idx+1)] = subset(alist, index[start:end])
    
    if retain_left and group_num * elem_num != len(index): 
        sub_lists['set'+str(idx+2)] = subset(alist, index[end:])
    
    return sub_lists


def draw_pr_roc(real_x, real_y, syn_data):    
    clf = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                  min_samples_leaf=10, random_state=42)
    
    train_x, test_x, train_y, test_y = train_test_split(real_x, real_y, test_size=0.1, random_state=0)

    #plot PR curve for synthetic test
    syn_test_index = syn_data.index
    syn_data_split = split_list(syn_test_index, group_num=5, shuffle=True, retain_left=False)
    
    #plot the roc curve
    probas_real = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_real[:, 1])
    roc_auc = auc(fpr, tpr)
    #print(syn_data_split)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    i = -1 
    color = ['r','b', 'y', 'g', 'c']
    for key, value in syn_data_split.items():
        #print(key)
        i = i + 1
        syn_test = syn_data.iloc[value]
        syn_test_x = syn_test[syn_test.columns[0:-1]]
        syn_test_y = syn_test[syn_test.columns[-1]]
        
        #calculate roc
        probas_syn = clf.fit(train_x, train_y).predict_proba(syn_test_x)
        fpr, tpr, thresholds = roc_curve(syn_test_y, probas_syn[:, 1])
        roc_auc = auc(fpr, tpr)
        ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=.8)
        #roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(fpr, tpr, label= 'Syn' + ' ' + key + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=2, alpha=.8)

        ax1.plot(fpr, tpr, label= 'Syn' + ' ' + key + ' ' + r'(AUC = %0.3f)' % (roc_auc), color=color[i], lw=2, alpha=.8)
        
        #calculate the precision and recall
        probas_syn = clf.fit(train_x, train_y).predict_proba(syn_test_x)[:, 1]
        precision, recall, thresholds = precision_recall_curve(syn_test_y, probas_syn)
        #calculate the precision score
        pr_auc = auc(recall, precision)
        
        #plt.plot([1, 0], [1, 0], linestyle='--', lw=1, alpha=.8)
        #pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(recall, precision, label= 'Syn' + ' ' + key + ' ' + r'(PRAUC = %0.3f)' % (pr_auc), lw=2, alpha=.8)

        ax2.plot(recall, precision, label= 'Syn' + ' ' + key + ' ' + r'(AUC = %0.3f)' % (pr_auc), color=color[i], lw=2, alpha=.8)
        
        
    plt.xlim([-0.00, 1.05])
    plt.ylim([-0.00, 1.05])
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    ax1.title.set_text('ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc="lower right", prop = {'size': 20})
    ax1.title.set_size(25)
    ax1.xaxis.label.set_size(20)
    ax1.yaxis.label.set_size(20)

    plt.xlim([-0.00, 1.05])
    plt.ylim([-0.00, 1.05])
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    ax2.title.set_text('PR Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='lower right', prop = {'size': 20})
    ax2.title.set_size(25)
    ax2.xaxis.label.set_size(20)
    ax2.yaxis.label.set_size(20)
    plt.savefig('result/figure/roc_pr.eps', dpi=300)

#cross-validation train + independent test data + external validation data to draw roc curve
def draw_roc_syn_test(real_x, real_y, syn_data, model):
    #np.random.seed(100)
    if model == 'lr':
        clf = LogisticRegression()
    elif model == 'rf':     
        clf = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                  min_samples_leaf=10, random_state=42)
    elif model == 'xgboost':  
        clf = xgb.XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=20, objective='binary:logistic', nthread=4, gamma=0, min_child_weight=5,
                                max_delta_step=0, subsample=0.8, colsample_bytree=0.7, colsample_bylevel=1, reg_alpha=1e-5, reg_lambda=1,
                                scale_pos_weight=1, seed=27, missing=None)
    else:
        print('error')
    
    train_x, test_x, train_y, test_y = train_test_split(real_x, real_y, test_size=0.1, random_state=0)

    #plot the roc curve
    probas_real = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_real[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    #plt.plot(fpr, tpr, label= 'Real set' + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=2, alpha=.8)
    
    #plot the roc curve for synthetic test set
    syn_test_index = syn_data.index
    syn_data_split = split_list(syn_test_index, group_num=5, shuffle=True, retain_left=False)
    #print(syn_data_split)
    for key, value in syn_data_split.items():
        #print(key)
        syn_test = syn_data.iloc[value]
        syn_test_x = syn_test[syn_test.columns[0:-1]]
        syn_test_y = syn_test[syn_test.columns[-1]]

        probas_syn = clf.fit(train_x, train_y).predict_proba(syn_test_x)
        # prediction_syn = clf.fit(train_x, train_y).predict(syn_test_x)
        # print(key)
        # print(accuracy_score(syn_test_y, prediction_syn))
        # print(precision_score(syn_test_y, prediction_syn))
        # print(recall_score(syn_test_y, prediction_syn))
        fpr, tpr, thresholds = roc_curve(syn_test_y, probas_syn[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=.8)
        plt.plot(fpr, tpr, label= 'Syn' + ' ' + key + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=2, alpha=.8)
    
    
    plt.xlim([-0.00, 1.05])
    plt.ylim([-0.00, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('result/figure/roc.eps', dpi=300)

def draw_pr_curve(real_x, real_y, syn_data):    
    clf = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                  min_samples_leaf=10, random_state=42)
    
    train_x, test_x, train_y, test_y = train_test_split(real_x, real_y, test_size=0.1, random_state=0)

    #plot PR curve for synthetic test
    syn_test_index = syn_data.index
    syn_data_split = split_list(syn_test_index, group_num=5, shuffle=True, retain_left=False)
    
    plt.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
    #print(syn_data_split)
    for key, value in syn_data_split.items():
        #print(key)
        syn_test = syn_data.iloc[value]
        syn_test_x = syn_test[syn_test.columns[0:-1]]
        syn_test_y = syn_test[syn_test.columns[-1]]
        

        
        #calculate the precision and recall
        probas_syn = clf.fit(train_x, train_y).predict_proba(syn_test_x)[:, 1]
        precision, recall, thresholds = precision_recall_curve(syn_test_y, probas_syn)
        #calculate the precision score
        pr_auc = auc(recall, precision)
        
        #plt.plot([1, 0], [1, 0], linestyle='--', lw=1, alpha=.8)
        plt.plot(recall, precision, label= 'Syn' + ' ' + key + ' ' + r'(PRAUC = %0.3f)' % (pr_auc), lw=2, alpha=.8)

    plt.xlim([-0.00, 1.05])
    plt.ylim([-0.00, 1.05])
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig('result/figure/pr.eps', dpi=300)


def merge_data(real_feature, real_label, syn_feature, syn_label):
    feature_combine = pd.concat([real_feature, syn_feature], axis=0)
    label_combine = pd.concat([real_label, syn_label], axis=0)
    return feature_combine, label_combine


######################################################################################################################################################


    

 
    