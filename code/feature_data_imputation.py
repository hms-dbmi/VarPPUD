#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:19:51 2021

@author: rayin
"""

import os, sys
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

case_gene_filter_labeled = pd.read_csv("data/processed/case_gene_filter_labeled.csv").iloc[:, 1:30]
column_name = list(case_gene_filter_labeled.columns.values)

#extract features
evolutionary_age = pd.read_csv("result/features/evolutionary_age.csv", index_col=0)
gene_dnds = pd.read_csv("result/features/gene_dnds.csv", index_col=0)
gene_essentiality = pd.read_csv("result/features/gene_essentiality.csv", index_col=0)
gene_haplo = pd.read_csv("result/features/gene_haplo.csv", index_col=0)
number_of_chem_interaction_action = pd.read_csv("result/features/number_of_chem_interaction_action.csv", index_col=0)
number_of_chem_interaction = pd.read_csv("result/features/number_of_chem_interaction.csv", index_col=0)
number_of_chem = pd.read_csv("result/features/number_of_chem.csv", index_col=0)
number_of_pathway = pd.read_csv("result/features/number_of_pathway.csv", index_col=0)
number_of_phenotype = pd.read_csv("result/features/number_of_phenotype.csv", index_col=0)
phylogenetic_number = pd.read_csv("result/features/phylogenetic_number.csv", index_col=0)

patient_id = [26, 381, 4599, 4875, 4919, 5225, 47, 428, 445, 994, 1046, 1047, 1297, 4028, 4053, 4065, 4080, 4172, 4193, 4216, 4227, 
              4244, 4291, 4329, 4470, 4503, 4552, 4555, 4557, 4563, 4591, 4637, 4661, 4663, 4699, 4777, 4814, 4839, 4903, 4938, 4977, 
              4990, 4998, 5088, 5106, 5166, 5260]


number_of_rare_disease = pd.read_csv("result/features/number_of_rare_disease.csv", index_col=0)
number_of_rare_disease = number_of_rare_disease[(~number_of_rare_disease['Patient ID'].isin(patient_id))].reset_index()
number_of_rare_disease = number_of_rare_disease['number_of_rare_disease']

number_of_total_disease = pd.read_csv("result/features/number_of_total_disease.csv", index_col=0)
number_of_total_disease = number_of_total_disease[(~number_of_total_disease['Patient ID'].isin(patient_id))].reset_index()
number_of_total_disease = number_of_total_disease['number_of_total_disease']

#concatenate gene based features
gene_feature = pd.concat([evolutionary_age, gene_dnds, gene_essentiality, gene_haplo,
                     number_of_chem_interaction_action, number_of_chem_interaction, number_of_chem,
                     number_of_pathway, number_of_phenotype, number_of_rare_disease,
                     number_of_total_disease, phylogenetic_number], axis=1)
gene_feature.to_csv('data/feature/gene_feature.csv')

# #this is the index with incorrect gene information
# outlier_index = [4, 55, 334, 408, 422, 496]

# def remove_outlier(df, index):
#     df_new = df.drop(labels=index, axis=0) #inplace=True
#     df_new.reset_index(drop=True, inplace=True)
#     return df_new
        
# case_gene_update = remove_outlier(case_gene_filter_labeled, outlier_index)
# evolutionary_age_update = remove_outlier(evolutionary_age, outlier_index)
# gene_dnds_update = remove_outlier(gene_dnds, outlier_index)
# gene_essentiality_update = remove_outlier(gene_essentiality, outlier_index)
# gene_haplo_update = remove_outlier(gene_haplo, outlier_index)
# number_of_chem_interaction_action_update = remove_outlier(number_of_chem_interaction_action, outlier_index)
# number_of_chem_interaction_update = remove_outlier(number_of_chem_interaction, outlier_index)
# number_of_chem_update = remove_outlier(number_of_chem, outlier_index)
# number_of_pathway_update = remove_outlier(number_of_pathway, outlier_index)
# number_of_phenotype_update = remove_outlier(number_of_phenotype, outlier_index)
# number_of_rare_disease_update = remove_outlier(number_of_rare_disease, outlier_index)
# number_of_total_disease_update = remove_outlier(number_of_total_disease, outlier_index)
# phylogenetic_number_update = remove_outlier(phylogenetic_number, outlier_index)

# feature = pd.concat([evolutionary_age_update, gene_dnds_update, gene_essentiality_update, gene_haplo_update,
#                      number_of_chem_interaction_action_update, number_of_chem_interaction_update, number_of_chem_update,
#                      number_of_pathway_update, number_of_phenotype_update, number_of_rare_disease_update,
#                      number_of_total_disease_update, phylogenetic_number_update], axis=1)

#feature.to_csv('gene_feature.csv')
#case_gene_update.to_csv('case_gene_update.csv')

def data_imputation(feature, category):
    #simple imputation, e.g., mean, zero
    if category == str('simple'):
        imputer = SimpleImputer(strategy='mean') #for median imputation replace 'mean' with 'median'
        imputer.fit(feature)
        feature_imputation = imputer.transform(feature)
    #fillin with the most freqency value
    if category == str('frequency'):
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(feature)
        feature_imputation = imputer.transform(feature)
    #knn tree
    if category == str('knn'):
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        feature_imputation = imputer.fit_transform(feature)
    #Multivariate Imputation by Chained Equation (MICE)
    if category == str('MICE'):
        imputer = IterativeImputer(random_state=0)
        imputer.fit(feature)
        feature_imputation = imputer.transform(feature)
    feature_imputation = pd.DataFrame(feature_imputation)
    feature_imputation.columns = feature.columns.values
    return feature_imputation


#################################################################################################################################################          
# #Imputation Using Deep Learning (Datawig)
# import datawig

# df_train, df_test = datawig.utils.random_split(train)

# #Initialize a SimpleImputer model
# imputer = datawig.SimpleImputer(
#     input_columns=['1','2','3','4','5','6','7', 'target'], # column(s) containing information about the column we want to impute
#     output_column= '0', # the column we'd like to impute values for
#     output_path = 'imputer_model' # stores model data and metrics
#     )

# #Fit an imputer model on the train data
# imputer.fit(train_df=df_train, num_epochs=50)

# #Impute missing values and return original dataframe with predictions
# imputed = imputer.predict(df_test)        
#################################################################################################################################################          
        
        
        