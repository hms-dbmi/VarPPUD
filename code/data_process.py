#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:38:54 2020

@author: rayin
"""

# pic-sure api lib
import PicSureHpdsLib
import PicSureClient

# python_lib for pic-sure
from python_lib.HPDS_connection_manager import tokenManager
from python_lib.utils import get_multiIndex_variablesDict

# analysis
import os, sys
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")


raw_data_all = pd.read_csv("data/raw/raw_data_all.csv")
DIDA_genes = pd.read_csv("data/raw/DIDA_genes.csv", sep='\t').iloc[:, 0:4]

#judge if any gene in DIDA can be found in UND
overlap_genes = []
for i in range(0, len(DIDA_genes)):
    for j in range(0, len(raw_data_all)):
        if DIDA_genes['Gene name'].iloc[i] == raw_data_all[raw_data_all.columns.values[21]].iloc[j]:
            overlap_genes.append(DIDA_genes['Gene name'].iloc[i])
            break;
            
#exclude the cases with missing values of gene names and variant interpretation
case_data_with_gene = []
case_data_without_gene = []
for i in range(0, len(raw_data_all)):
    if pd.isna(raw_data_all[raw_data_all.columns.values[21]].iloc[i]) or pd.isna(raw_data_all[raw_data_all.columns.values[26]].iloc[i]):
        case_data_without_gene.append(raw_data_all.iloc[i])
    else:
        case_data_with_gene.append(raw_data_all.iloc[i])
        
case_data_with_gene = pd.DataFrame(case_data_with_gene).reset_index()
case_data_with_gene = case_data_with_gene.iloc[:, 1:39]
case_data_without_gene = pd.DataFrame(case_data_without_gene).reset_index()
case_data_without_gene = case_data_without_gene.iloc[:, 1:39]


#filter the samples by row, axis=0 delete row and by column, axis = 1 delete column
def data_filter(df):
    row_list = []
    row_count = df.shape[1]
    for i in range(0, df.shape[0]):
        if df.iloc[i].isna().sum() > row_count/(2/3): 
            print(i)
            row_list.append(i)
        
    df_delete_row = df.drop(labels=row_list, axis=0) #inplace=True
    df_delete_row.reset_index(drop=True, inplace=True)


    column_count = df_delete_row.shape[0]
    column_list = []
    for j in range(0, df_delete_row.shape[1]):
        if df_delete_row[df_delete_row.columns.values[j]].isna().sum() > column_count/2:
            column_list.append(j)

    drop_column = [] 
    for i in range(0, len(column_list)):
        drop_column.append(df_delete_row.columns.values[column_list[i]])
    
    df_filter = df_delete_row.drop(labels=drop_column, axis=1)
    return(df_filter)

case_data_with_gene_filter = data_filter(case_data_with_gene)

und_all_gene = pd.read_csv('data/raw/udn_gene_count.csv')
und_all_gene = und_all_gene.iloc[:, 1:3]

for i in range(0, len(case_data_with_gene_filter)):
    for j in range(0, len(und_all_gene)):
        if case_data_with_gene_filter[case_data_with_gene_filter.columns.values[15]].iloc[i] == und_all_gene['gene'].iloc[j]:
            break;
        elif j == len(und_all_gene) - 1:
            print(i)
            print(case_data_with_gene_filter[case_data_with_gene_filter.columns.values[15]].iloc[i])


#statistics and visualization
column_name = list(case_data_with_gene_filter.columns.values)
case_data_with_gene_filter[column_name[2]].describe()

#column 5-8, 15, 16, 18, 19, 20, 25, 28
Counter(case_data_with_gene_filter[column_name[20]])


case_gene_filter_labeled = case_data_with_gene_filter[case_data_with_gene_filter['\\11_Candidate genes\\Status\\'] != 'rejected']
case_gene_filter_labeled = case_gene_filter_labeled[case_gene_filter_labeled['\\12_Candidate variants\\03 Interpretation\\'] != 'investigation_n']

case_gene_filter_labeled = case_gene_filter_labeled.replace('benign', 'less_pathogenic')
case_gene_filter_labeled = case_gene_filter_labeled.replace('likely_benign', 'less_pathogenic')
case_gene_filter_labeled = case_gene_filter_labeled.replace('variant_u_s', 'less_pathogenic')
#case_gene_filter_labeled = case_gene_filter_labeled.replace('investigation_n', 'less_pathogenic')
case_gene_filter_labeled = case_gene_filter_labeled.replace('likely_pathogenic', 'pathogenic')
case_gene_filter_labeled.to_csv("data/processed/case_gene_filter_labeled.csv")

#statistics for pathogenic and non-pathogenic cases
def characteristics_percentage(df, column_index):
    charac = df.groupby([column_name[column_index], column_name[20]])
    charac = pd.DataFrame(charac.size())
    for i in range(0, len(charac)):
        percentage = charac.iloc[i]/charac.sum()
        print(charac.iloc[i])
        print(percentage)
        print('\t')
    
    pathogen_unknown = 0
    nonpathogen_unknown = 0
    for j in range(0, len(df)):
        if pd.isnull(df[column_name[column_index]].iloc[j]):
            if df[column_name[20]].iloc[j] == 'pathogenic':
                pathogen_unknown = pathogen_unknown + 1
            elif df[column_name[20]].iloc[j] == 'less_pathogenic':
                nonpathogen_unknown = nonpathogen_unknown + 1
    nonpathogen_unknown_perct = nonpathogen_unknown/charac.sum()
    pathogen_unknown_perct = pathogen_unknown/charac.sum()
    print("nonpathogenic unknown:")
    print('count:', nonpathogen_unknown)
    print('percentage:', nonpathogen_unknown_perct)
    print('\t')

    print("pathogenic unknown:")
    print('count:', pathogen_unknown)
    print('percentage:', pathogen_unknown_perct)
    print('\t')

#characteristics_percentage(case_gene_filter_labeled, 5)
#characteristics_percentage(case_gene_filter_labeled, 6)
#characteristics_percentage(case_gene_filter_labeled, 7)
#characteristics_percentage(case_gene_filter_labeled, 8)


#statistics for patient current age
case_gene_filter_labeled[case_gene_filter_labeled[column_name[20]] == 'pathogenic'][column_name[4]].describe()
case_gene_filter_labeled[case_gene_filter_labeled[column_name[20]] == 'less_pathogenic'][column_name[4]].describe()


#statistics of age for the first symptom onset at different age groups 
def age_symptom_onset(df, column_index):
    df_len = len(df)
    
    #statistics for pathogenic cases
    pathogen_age = case_gene_filter_labeled[case_gene_filter_labeled[column_name[20]] == 'pathogenic'][column_name[column_index]]
    pathogen_age = pd.DataFrame(pathogen_age)
    
    pathogen_age_1to10 = np.sum(list(map(lambda x: x >= 0 and x <= 10, pathogen_age[column_name[column_index]])))
    print("Pathogenic - number of age group 0-10:", pathogen_age_1to10)
    print('percentage:', pathogen_age_1to10/df_len)
    print('\t')
    
    pathogen_age_10to30 = np.sum(list(map(lambda x: x > 10 and x <= 30, pathogen_age[column_name[column_index]])))
    print("Pathogenic - number of age group 10-30:", pathogen_age_10to30)
    print('percentage:', pathogen_age_10to30/df_len)
    print('\t')
    
    pathogen_age_30to60 = np.sum(list(map(lambda x: x > 30 and x <= 60, pathogen_age[column_name[column_index]])))
    print("Pathogenic - number of age group 30-60:", pathogen_age_30to60)
    print('percentage:', pathogen_age_30to60/df_len)
    print('\t')
    
    pathogen_age_60above = np.sum(list(map(lambda x: x > 60 , pathogen_age[column_name[column_index]])))
    print("Pathogenic - number of age group > 60:", pathogen_age_60above)
    print('percentage:', pathogen_age_60above/df_len)
    print('\t')

    #statistics for non-pathogenic cases
    nonpathogen_age = case_gene_filter_labeled[case_gene_filter_labeled[column_name[20]] == 'less_pathogenic'][column_name[column_index]]
    nonpathogen_age = pd.DataFrame(nonpathogen_age)
    
    nonpathogen_age_1to10 = np.sum(list(map(lambda x: x >= 0 and x <= 10, nonpathogen_age[column_name[column_index]])))
    print("less pathogenic - number of age group 0-10:", nonpathogen_age_1to10)
    print('percentage:', nonpathogen_age_1to10/df_len)
    print('\t')
    
    nonpathogen_age_10to30 = np.sum(list(map(lambda x: x > 10 and x <= 30, nonpathogen_age[column_name[column_index]])))
    print("less pathogenic - number of age group 10-30:", nonpathogen_age_10to30)
    print('percentage:', nonpathogen_age_10to30/df_len)
    print('\t')
    
    nonpathogen_age_30to60 = np.sum(list(map(lambda x: x > 30 and x <= 60, nonpathogen_age[column_name[column_index]])))
    print("Less pathogenic - number of age group 30-60:", nonpathogen_age_30to60)
    print('percentage:', nonpathogen_age_30to60/df_len)
    print('\t')
    
    nonpathogen_age_60above = np.sum(list(map(lambda x: x > 60 , nonpathogen_age[column_name[column_index]])))
    print("Less pathogenic - number of age group > 60:", nonpathogen_age_60above)
    print('percentage:', nonpathogen_age_60above/df_len)
    print('\t')
    
    #check if there is nan value
    pathogen_age_unknown = 0
    nonpathogen_age_unknown = 0
    for j in range(0, len(df)):
        if pd.isna(df[column_name[column_index]].iloc[j]):
            if df[column_name[20]].iloc[j] == 'pathogenic':
                pathogen_age_unknown = pathogen_age_unknown + 1
            elif df[column_name[20]].iloc[j] == 'less_pathogenic':
                nonpathogen_age_unknown = nonpathogen_age_unknown + 1
    nonpathogen_age_unknown_perct = nonpathogen_age_unknown/df_len
    pathogen_age_unknown_perct = pathogen_age_unknown/df_len
    print("nonpathogenic age unknown:")
    print('count:', nonpathogen_age_unknown)
    print('percentage:', nonpathogen_age_unknown_perct)
    print('\t')

    print("pathogenic age unknown:")
    print('count:', pathogen_age_unknown)
    print('percentage:', pathogen_age_unknown_perct)
    print('\t')     
        
#age_symptom_onset(case_gene_filter_labeled, 3)




