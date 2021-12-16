#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 22:05:18 2021

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
import seaborn as sns

from matplotlib import rcParams

from model import draw_roc_curve
from model import patient_variable_influence
from model import draw_roc_train_test_split
from model import randomforest_base
from model import xgboost_base
from model import draw_roc_syn_test
from model import draw_roc_curve
from feature_data_imputation import data_imputation

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")
sys.path.append(os.path.abspath("/Users/rayin/Google Drive/Harvard/5_data/UDN/work/code/"))

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



##################################################################################################################################
# #feature correlation heatmap
# plt.figure(figsize=(16, 16))
# # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
# heatmap = sns.heatmap(feature.corr(), vmin=-1, vmax=1, annot=False)
# # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
# heatmap.set_xticklabels(
#     heatmap.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
# plt.savefig("result/figure/heatmap.png", bbox_inches='tight', dpi=600)



##################################################################################################################################
#interpretation: shap values
np.random.seed(0)

train_x, test_x, train_y, test_y = train_test_split(feature_imputation, label, test_size = 0.2)
model = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                          min_samples_leaf=10, random_state=42)
model.fit(train_x, train_y)

shap_values = shap.TreeExplainer(model).shap_values(test_x)
shap.plots.force(shap_values[1])

#shap.summary_plot(shap_values, train_x, plot_type="bar")
f = plt.figure(figsize=(16, 8))
shap.summary_plot(shap_values[1], test_x)  
#f.savefig("result/figure/shap.eps", bbox_inches='tight', dpi=600)

#variable importance bar plot (shape and random forest)
shap.summary_plot(shap_values[0], test_x, plot_type="bar", max_display=20)

fig = plt.figure(figsize=(12, 16))
feature_names = feature.columns.values

# sorted_idx = model.feature_importances_.argsort()
# plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx], color='g')
# fig.xlabel("Random Forest Feature Importance")

#random forest model 
result = permutation_importance(
    model, test_x, test_y, random_state=42, n_jobs=1)
result_mean = result.importances_mean
sorted_idx = np.argsort(result_mean, axis=-1)
plt.barh(feature_names[sorted_idx[18:38]], result_mean[sorted_idx[18:38]], color='g')
plt.xlabel("Random Forest Feature Importance", size=20)
#fig.set_xlabel(fontsize=15)
plt.tick_params(labelsize=20)
plt.xlim(0, 0.04)

#shap interaction values
f = plt.figure(figsize=(20, 20))
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(test_x)
shap.summary_plot(shap_interaction_values[0], test_x, max_display=20)
f.savefig("result/figure/shap_interaction.png", bbox_inches='tight', dpi=100)

#independence 
shap.dependence_plot("number of phenotypes", shap_values[1], test_x, interaction_index="number of total diseases")

f = plt.figure()
shap.summary_plot(shap_values, test_x)
f.savefig("result/figure/summary_plot1.png", bbox_inches='tight', dpi=600)


##################################################################################################################################
#measure the contribution adding demographic information of patient for pathogenicity prediction
raw_auc = pd.read_csv('result/auc_difference.csv', index_col=0)
auc_index = raw_auc.index.values
auc_columns = raw_auc.columns.values
age_symptom_onset = []
current_age = []
ethnicity = []
gender = []
race = []

for i in range(0, raw_auc.shape[1]):
    age_symptom_onset.append(round(raw_auc[auc_columns[i]].iloc[1] - raw_auc[auc_columns[i]].iloc[0], 4))
    current_age.append(round(raw_auc[auc_columns[i]].iloc[2] - raw_auc[auc_columns[i]].iloc[0], 4))
    ethnicity.append(round(raw_auc[auc_columns[i]].iloc[3] - raw_auc[auc_columns[i]].iloc[0], 4))
    gender.append(round(raw_auc[auc_columns[i]].iloc[4] - raw_auc[auc_columns[i]].iloc[0], 4))
    race.append(round(raw_auc[auc_columns[i]].iloc[5] - raw_auc[auc_columns[i]].iloc[0], 4))

attri_diff = pd.concat([pd.Series(age_symptom_onset), pd.Series(current_age), pd.Series(ethnicity), 
                        pd.Series(gender), pd.Series(race)], axis=1)


f = plt.figure(figsize=(10,5))
#plt.title('Examples of boxplot',fontsize=20)
labels = 'age symptom onset','current age','ethnicity','gender', 'race'
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
plt.ylabel(r'$\Delta$AUC')
fig = plt.boxplot([age_symptom_onset, current_age, ethnicity, gender, race], labels = labels, showmeans=True,
            patch_artist=True)
color=['pink', 'lightblue', 'lightgreen', 'lightcyan', 'tomato']
for box, color in zip(fig['boxes'], color):
    box.set_facecolor(color)
for median in fig['medians']:
    median.set(color='black')
f.savefig("result/figure/DeltaAUC.eps", bbox_inches='tight', dpi=300)
f.savefig("result/figure/DeltaAUC.png", bbox_inches='tight', dpi=300)


##################################################################################################################################
#plot predictive accuracy with different lengths of features
feature_acc = []
feature_pre = []
feature_rec = []
feature_f1 = []
feature_mcc = []
feature_auc = []
train_x, test_x, train_y, test_y = train_test_split(feature_imputation, label, test_size=0.1, random_state=42)
for i in range(0, 38):  
    clf = ensemble.RandomForestClassifier(n_estimators=340, max_depth=5, max_features=7, min_samples_split=30,
                                  min_samples_leaf=10, random_state=42)
    kfold = StratifiedKFold(n_splits=5) #shuffle=True
    #kfold = KFold(n_splits=5, shuffle=True)

    clf = ensemble.RandomForestClassifier().fit(train_x.iloc[:, 0:i+1], train_y)
    test_acc = accuracy_score(test_y, clf.predict(test_x.iloc[:, 0:i+1]))
    test_pre = precision_score(test_y, clf.predict(test_x.iloc[:, 0:i+1]))
    test_rec = recall_score(test_y, clf.predict(test_x.iloc[:, 0:i+1]))
    test_f1 = f1_score(test_y, clf.predict(test_x.iloc[:, 0:i+1]))
    test_mcc = matthews_corrcoef(test_y, clf.predict(test_x.iloc[:, 0:i+1]))
    test_auc = roc_auc_score(test_y, clf.predict(test_x.iloc[:, 0:i+1]))
    
    feature_acc.append(test_acc)
    feature_pre.append(test_pre)
    feature_rec.append(test_rec)
    feature_f1.append(test_f1)
    feature_mcc.append(test_mcc)
    feature_auc.append(test_auc)
    
x=np.arange(1,39)
l1=plt.plot(x,feature_acc,'r-',label='Accuracy')
l2=plt.plot(x,feature_pre,'g-',label='Precision')
l3=plt.plot(x,feature_rec,'b-',label='Recall')
l4=plt.plot(x,feature_f1,'y-',label='F-score')
l5=plt.plot(x,feature_auc,'r-',label='AUROC')
#plt.title('The Lasers in Three Conditions')
plt.xlabel('Number of features')
plt.ylabel('Performance')
plt.xlim(0, 40)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.savefig("result/figure/feature_performance.png", bbox_inches='tight', dpi=300)

#plt.show()
























