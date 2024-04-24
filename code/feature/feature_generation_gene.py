#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:20:03 2021

@author: rayin
"""

# analysis
import os, sys
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter


os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

def isnan(df):
    count = 0
    if isinstance(df, list):
        for i in range(0, len(df)):
            if str(df[i]).lower() == 'nan':
                count = count + 1
            else:
                pass
    elif isinstance(df, pd.DataFrame):
        column_value = df.columns[0]
        for i in range(0, len(df)):
            if str(df[column_value].iloc[i]).lower() == 'nan':
                count = count + 1
            else:
                pass
    return count

case_gene_filter_labeled = pd.read_csv("data/processed/variant_clean.csv", index_col=0)
#column_name = list(case_gene_filter_labeled.columns.values)

########################################################################################################################################################
#feature generation based on genes
#the number of pathways of genes
gene_pathway_data = pd.read_csv("data/database/gene_pathway.csv", sep ='\t').iloc[:, 0:4]
number_of_pathway = []

for i in range(0, len(case_gene_filter_labeled)):
    count = 0
    for j in range(0, len(gene_pathway_data)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_pathway_data["GeneSymbol"].iloc[j]).upper():
            count = count + 1
    number_of_pathway.append(count)        
#pd.DataFrame(number_of_pathway).to_csv("number_of_pathway.csv")     

########################################################################################################################################################
#http://ctdbase.org/downloads/
#the number of chemical and interaction of genes
gene_chem_data = pd.read_csv("data/database/gene_chem.csv")
gene_chem_data = gene_chem_data[gene_chem_data['Organism'] == 'Homo sapiens']  
#chemical-gene dataset
gene_chem = gene_chem_data[['ChemicalName', 'GeneSymbol']]
gene_chem = gene_chem.drop_duplicates().reset_index(drop=True)
#chemical-gene interaction dataset
gene_chem_interaction = gene_chem_data[['Interaction', 'GeneSymbol']]
gene_chem_interaction = gene_chem_interaction.drop_duplicates().reset_index(drop=True)
#chemical-gene interaction action dataset
gene_chem_interaction_action = gene_chem_data[['InteractionActions', 'GeneSymbol']]
gene_chem_interaction_action = gene_chem_interaction_action.drop_duplicates().reset_index(drop=True)

number_of_chem = []
number_of_chem_interaction = []
number_of_chem_interaction_action = []

for i in range(0, len(case_gene_filter_labeled)):
    count = 0
    for j in range(0, len(gene_chem)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_chem["GeneSymbol"].iloc[j]).upper():
            count = count + 1
    number_of_chem.append(count) 
#pd.DataFrame(number_of_chem).to_csv("number_of_chem.csv")

for i in range(0, len(case_gene_filter_labeled)):
    count = 0
    for j in range(0, len(gene_chem_interaction)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_chem_interaction["GeneSymbol"].iloc[j]).upper():
            count = count + 1
    number_of_chem_interaction.append(count)   
#pd.DataFrame(number_of_chem_interaction).to_csv("number_of_chem_interaction.csv")

for i in range(0, len(case_gene_filter_labeled)):
    count = 0
    for j in range(0, len(gene_chem_interaction_action)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_chem_interaction_action["GeneSymbol"].iloc[j]).upper():
            count = count + 1
    number_of_chem_interaction_action.append(count) 
#pd.DataFrame(number_of_chem_interaction_action).to_csv("number_of_chem_interaction_action.csv")

########################################################################################################################################################
#the number of phenotype of genes 
#https://hpo.jax.org/app/download/annotation   
gene_phenotype_data = pd.read_csv("data/database/gene_phenotype.csv", sep ='\t')
gene_phenotype_data = gene_phenotype_data[['HPO label', 'entrez-gene-symbol']]
gene_phenotype_data = gene_phenotype_data.drop_duplicates().reset_index(drop=True)

number_of_phenotype = []

for i in range(0, len(case_gene_filter_labeled)):
    count = 0
    for j in range(0, len(gene_phenotype_data)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_phenotype_data["entrez-gene-symbol"].iloc[j]).upper():
            count = count + 1
    number_of_phenotype.append(count) 
#pd.DataFrame(number_of_phenotype).to_csv("number_of_phenotype.csv")

# ########################################################################################################################################################
# #the number of disease of genes, this has been replaced by the total number of diseases
# #http://ctdbase.org/downloads/;jsessionid=E262DF92313F378B91AFAA739D71F41F#gd
# import csv
# import gzip

# gene_disease_data = []
# with open("/Users/rayin/Downloads/gene_disease_data.csv", "r") as f:
#      reader = csv.reader(f)
#      for row in reader:
#         gene_disease_data.append(row[0:3])        

# gene_disease_data = pd.DataFrame(gene_disease_data, columns = ['GeneSymbol','GeneID','DiseaseName'])
# gene_disease = gene_disease_data.drop_duplicates().reset_index(drop=True)

# gene_symbol_unique = list(gene_disease['GeneSymbol'].unique())

# len(gene_disease[gene_disease['GeneSymbol'] == 'SGCE'])


# number_of_disease = []
# for i in range(0, len(case_gene_filter_labeled)):
#     count = 0
#     for j in range(0, len(gene_disease)):
#         if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_disease["GeneSymbol"].iloc[j]).upper():
#             count = count + 1
#     number_of_disease.append(count) 
    
########################################################################################################################################################
#https://proteinhistorian.docpollard.org/
#evolutionary age of genes (ProteinHistorian)
candidate_gene = case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\']

# with open("candidate_gene.txt", 'w') as f:
#     for s in candidate_gene:
#         f.write(str(s) + '\n')
evolutionary_age_data = pd.read_csv('data/database/evolutionary_age.csv', sep ='\t')
phylogenetic_number_data = pd.read_csv('data/database/phylogenetic_number.csv', sep ='\t')

evolutionary_age = []
phylogenetic_number = []

for i in range(0, len(case_gene_filter_labeled)):
    evolution_flag = 0
    phylogenetic_flag = 0
    for j in range(0, len(evolutionary_age_data)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() in str(evolutionary_age_data["protein"].iloc[j]).upper():
            evolutionary_age.append(evolutionary_age_data["age"].iloc[j])
            evolution_flag = 1
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() in str(phylogenetic_number_data["protein"].iloc[j]).upper():
            phylogenetic_number.append(phylogenetic_number_data["HUMAN"].iloc[j])
            phylogenetic_flag = 1
            break
    if evolution_flag == 0:
        evolutionary_age.append('nan')
    if phylogenetic_flag == 0:
        phylogenetic_number.append('nan')
#pd.DataFrame(evolutionary_age).to_csv('evolutionary_age.csv')
#pd.DataFrame(phylogenetic_number).to_csv('phylogenetic_number.csv')

######################################################################################################################################################## 
#gene essentiality
#https://v3.ogee.info/#/home
gene_essentiality_data = pd.read_csv('data/database/gene_essentiality_data.csv', sep=',')
gene_essentiality_data['symbols'].isnull().sum()

gene_essentiality = []
for i in range(0, len(case_gene_filter_labeled)):
    flag = 0
    for j in range(0, len(gene_essentiality_data)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_essentiality_data["symbols"].iloc[j]).upper():
            flag = 1
            if gene_essentiality_data["essentiality consensus"].iloc[j] == str('Nonessential'):
                gene_essentiality.append(int(0))
            elif gene_essentiality_data["essentiality consensus"].iloc[j] == str('Conditional'):
                gene_essentiality.append(int(1))
            elif gene_essentiality_data["essentiality consensus"].iloc[j] == str('Essential'):
                gene_essentiality.append(int(2))
            else:
                print('error')
            break
        
    if flag == 0:
        gene_essentiality.append('nan')
    elif flag == 1:
        pass
    
#pd.DataFrame(gene_essentiality).to_csv('gene_essentiality.csv')

######################################################################################################################################################## 
#dn/ds of genes
#http://www.h-invitational.jp/evola/download.html
gene_dnds_data = pd.read_csv('data/database/dnds.csv', sep='\t')
gene_dnds_accession = pd.read_csv('data/database/dnds_accession.csv', sep='\t')

#gene_dnds_data = gene_dnds_data[gene_dnds_data['Species.2'] == 'Chimpanzee']
#gene_dnds_accession = gene_dnds_accession[gene_dnds_accession['other species name'] == 'Chimpanzee']

human_id = []
for i in range(0, len(case_gene_filter_labeled)):
    flag = 0
    for j in range(0, len(gene_dnds_accession)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_dnds_accession["human gene symbol"].iloc[j]).upper():
            flag = 1
            human_id.append(gene_dnds_accession["H-Inv human transcript ID"].iloc[j])
            break
    if flag == 0:
        human_id.append('nan')
    elif flag == 1:
        pass

for j in range(0, len(gene_dnds_data)):
    if gene_dnds_data["dN"].iloc[j] == str('-'):
        gene_dnds_data["dN"].iloc[j] = str('0')
    if gene_dnds_data["dS"].iloc[j] == str('-'):
        gene_dnds_data["dS"].iloc[j] = str('0')
        
gene_dnds_data["dN"] = pd.to_numeric(gene_dnds_data["dN"])
gene_dnds_data["dS"] = pd.to_numeric(gene_dnds_data["dS"])

gene_dnds_result = []
gene_dnds = []
for i in range(0, len(human_id)):
    species = 0
    dnds = 0
    result = 0
    flag = 0
    for j in range(0, len(gene_dnds_data)):
        if str(human_id[i]).upper() == str(gene_dnds_data["Seq.1"].iloc[j]).upper():
            flag = 1
            dn = gene_dnds_data["dN"].iloc[j]
            ds = gene_dnds_data["dS"].iloc[j]
            if float(dn) == 0 or float(ds) == 0:
                result = 0
            elif float(dn) != 0 and float(ds) != 0:
                result = float(dn)/float(ds)
        species = species + 1
        dnds = dnds + result
    if flag == 0:
        gene_dnds.append('nan')
    elif flag == 1:
        gene_dnds.append(dnds/species)
    
#pd.DataFrame(gene_dnds).to_csv('gene_dnds.csv')   


######################################################################################################################################################## 
#loss of function intolerance score of genes
gene_intolerance_data = pd.read_csv('data/database/gene_intolerance.csv', sep='\t')

gene_intolerance = []
for i in range(0, len(case_gene_filter_labeled)):
    flag = 0
    for j in range(0, len(gene_intolerance_data)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() == str(gene_intolerance_data["Gene"].iloc[j]).upper():
            flag = 1
            gene_intolerance.append(gene_intolerance_data["HC LoF"].iloc[j])
            break
    if flag == 0:
        gene_intolerance.append('nan')
    elif flag == 1:
        pass
#pd.DataFrame(gene_intolerance).to_csv('gene_intolerance.csv')

######################################################################################################################################################## 
#https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1001154
#haploinsufficiency score of genes
gene_haplo_data = pd.read_csv('data/database/gene_haploinsufficient.csv', sep='\t')

gene_haplo = []
for i in range(0, len(case_gene_filter_labeled)):
    flag = 0
    for j in range(0, len(gene_haplo_data)):
        if str(case_gene_filter_labeled['\\11_Candidate genes\\Gene Name\\'].iloc[i]).upper() in str(gene_haplo_data["gene"].iloc[j]).upper():
            flag = 1
            gene_haplo.append(gene_haplo_data["probability"].iloc[j])
            break
    if flag == 0:
        gene_haplo.append('nan')
    elif flag == 1:
        pass
#pd.DataFrame(gene_haplo).to_csv('gene_haplo.csv')

#number of disease regarding to genes, this is directly obtained from Genecards through mapping
#https://www.genecards.org/
number_of_rare_disease = pd.read_csv("result/features/number_of_rare_disease.csv", index_col=0)
number_of_total_disease = pd.read_csv("result/features/number_of_total_disease.csv", index_col=0)

#concatenate gene based features
gene_feature = pd.concat([evolutionary_age, gene_dnds, gene_essentiality, gene_haplo,
                     number_of_chem_interaction_action, number_of_chem_interaction, number_of_chem,
                     number_of_pathway, number_of_phenotype, number_of_rare_disease,
                     number_of_total_disease, phylogenetic_number], axis=1)
#gene_feature.to_csv('data/feature/gene_feature.csv')

########################################################################################################################################################
#patient level variables
case_gene_filter_labeled = pd.read_csv("data/processed/case_gene_filter_labeled.csv", index_col=0)
case_gene_filter_labeled.reset_index(inplace=True, drop=True)
column_name = list(case_gene_filter_labeled.columns.values)

patient_level_variable = case_gene_filter_labeled[[column_name[3], column_name[4], column_name[5], column_name[6], column_name[7]]]
patient_level_variable.columns = ['age_symptom_onset', 'current_age', 'ethnicity', 'gender', 'race']

#remove the unqualified cases from raw data and extract demographic information. The patiant id to be eliminated
patient_id = [26, 381, 4599, 4875, 4919, 5225, 47, 428, 445, 994, 1046, 1047, 1297, 4028, 4053, 4065, 4080, 4172, 4193, 4216, 4227, 
              4244, 4291, 4329, 4470, 4503, 4552, 4555, 4557, 4563, 4591, 4637, 4661, 4663, 4699, 4777, 4814, 4839, 4903, 4938, 4977, 
              4990, 4998, 5088, 5106, 5166, 5260]

outlier_case = case_gene_filter_labeled[(case_gene_filter_labeled['Patient ID'].isin(patient_id))]
outlier_index = list(outlier_case.index)

def remove_outlier(df, index):
    df_new = df.drop(labels=index, axis=0) #inplace=True
    df_new.reset_index(drop=True, inplace=True)
    return df_new

patient_level_variable = remove_outlier(patient_level_variable, outlier_index)
patient_level_variable['race'].replace(float('nan'), 'unknown', inplace=True)
patient_level_variable['ethnicity'].replace(float('nan'), 'unknown', inplace=True)

patient_level_variable_onehot = pd.get_dummies(patient_level_variable)
#patient_level_variable_onehot.to_csv('data/feature/patient_level_variable_onehot.csv')





    
