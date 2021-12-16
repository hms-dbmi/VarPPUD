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
# https://github.com/hms-dbmi/Access-to-Data-using-PIC-SURE-API/tree/master/NIH_Undiagnosed_Diseases_Network
from python_lib.HPDS_connection_manager import tokenManager
from python_lib.utils import get_multiIndex_variablesDict

# analysis
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

#loading raw input patient data extracted by PIC-SURE from UDN 
raw_data_all = pd.read_csv("data/raw/raw_data_all.csv")

#inclusion criteria       
#exclude the cases with missing values of candidate gene and variant interpretation
case_data_with_gene = []
case_data_without_gene = []
for i in range(0, len(raw_data_all)):
    if pd.isna(raw_data_all[raw_data_all.columns.values[21]].iloc[i]) or pd.isna(raw_data_all[raw_data_all.columns.values[26]].iloc[i]):
        case_data_without_gene.append(raw_data_all.iloc[i])
    else:
        case_data_with_gene.append(raw_data_all.iloc[i])

#reformat        
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


#statistics and visualization
column_name = list(case_data_with_gene_filter.columns.values)
case_data_with_gene_filter[column_name[2]].describe()

#Variant interpretation. Remove the rejected and under investigation cases.
Counter(case_data_with_gene_filter[column_name[20]])
case_gene_filter_labeled = case_data_with_gene_filter[case_data_with_gene_filter['\\11_Candidate genes\\Status\\'] != 'rejected']
case_gene_filter_labeled = case_gene_filter_labeled[case_gene_filter_labeled['\\12_Candidate variants\\03 Interpretation\\'] != 'investigation_n']

#define 'benign', 'likely benign' and 'uncertain' as 'less pathogenic', 'likely pathogenic' and 'pathogenic' as pathogenic'.
case_gene_filter_labeled = case_gene_filter_labeled.replace('benign', 'less_pathogenic')
case_gene_filter_labeled = case_gene_filter_labeled.replace('likely_benign', 'less_pathogenic')
case_gene_filter_labeled = case_gene_filter_labeled.replace('variant_u_s', 'less_pathogenic')
#case_gene_filter_labeled = case_gene_filter_labeled.replace('investigation_n', 'less_pathogenic')
case_gene_filter_labeled = case_gene_filter_labeled.replace('likely_pathogenic', 'pathogenic')
case_gene_filter_labeled.to_csv("data/processed/case_gene_filter_labeled.csv")  #521 cases

#Manually remove the cases with unknown or incorrect gene names ('Exon-level microarray', '22q11.2 FISH', '20p13 duplication', etc.) and 
#6 cases are excluded (index (after index_reset): 4, 55, 334, 408, 422, 496)
#Loading cases after manual curation from file case_gene_update.csv'
case_gene_update = pd.read_csv('data/processed/case_gene_update.csv', index_col=0) #515 cases
column_name = list(case_gene_update.columns.values)
protein_var = case_gene_update['\\12_Candidate variants\\09 Protein\\']

#Manual curation to remove cases with missing candidate variants or complex variants (e.g., long deletion and duplication) 
#Export a clean version named  'variant_clean.csv'




























