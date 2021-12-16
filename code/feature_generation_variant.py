#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:45:28 2021

@author: rayin
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import random

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

case_gene_update = pd.read_csv("data/processed/variant_clean.csv", index_col=0)

#mutation_taster
mutation_taster = pd.read_excel("data/score/mutation_taster.xlsx")
mutation_taster = mutation_taster.drop(mutation_taster[mutation_taster['genesymbol'] == 'compound'].index)

mutation_taster.drop_duplicates(subset='position', keep='first', inplace=True)

mutation_taste_main = []
missing_case = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(mutation_taster)):    
        if case_gene_update['START'].iloc[i] == mutation_taster['position'].iloc[j]:
            if case_gene_update['\\11_Candidate genes\\Gene Name\\'].iloc[i] == mutation_taster['genesymbol'].iloc[j]:
                flag = 1
                mutation_taste_main.append(mutation_taster.iloc[j])
    if flag == 0:
        missing_case.append(case_gene_update.iloc[i])
          
mutation_taste_main =  pd.DataFrame(mutation_taste_main)
missing_case = pd.DataFrame(missing_case)

mutation_taste_sup = pd.read_excel("data/score/mutation_taster_sup.xlsx")
mutation_taste_merge = pd.concat([mutation_taste_main, mutation_taste_sup])


############################################################################################################################################
cadd = pd.read_csv("data/score/cadd_score.tsv")
fathmm = pd.read_csv("data/score/fathmm_score.csv")
proven = pd.read_csv("data/score/PROVEN_score.tsv")
vest = pd.read_excel("data/score/vest_score.xlsx")

#cadd score
cadd_score = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(cadd)):
        if case_gene_update['START'].iloc[i] == cadd['POS'].iloc[j]:
            flag = 1
            break;
    if flag == 1:
        cadd_score.append(cadd['RawScore'].iloc[j])
    elif flag == 0:
        cadd_score.append('nan')

#fathms score
fathmm_score = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(fathmm)):
        if case_gene_update['START'].iloc[i] == fathmm['Position'].iloc[j]:
            flag = 1
            break;
    if flag == 1:
        fathmm_score.append(fathmm['Coding Score'].iloc[j])
    elif flag == 0:
        fathmm_score.append('nan')

#vest score
vest_score = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(vest)):
        if case_gene_update['START'].iloc[i] == vest['Position'].iloc[j]:
            flag = 1
            break;
    if flag == 1:
        vest_score.append(vest['VEST score'].iloc[j])
    elif flag == 0:
        vest_score.append('nan')


#proven score
proven_score = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(proven)):
        if case_gene_update['START'].iloc[i] == proven['Pos'].iloc[j]:
            flag = 1
            break;
    if flag == 1:
        proven_score.append(proven['SCORE'].iloc[j])
    elif flag == 0:
        proven_score.append('nan')


#SIFT score
sift_score = []
for i in range(0, len(case_gene_update)):
    flag = 0
    for j in range(0, len(proven)):
        if case_gene_update['START'].iloc[i] == proven['Pos'].iloc[j]:
            flag = 1
            break;
    if flag == 1:
        sift_score.append(proven['SCORE.1'].iloc[j])
    elif flag == 0:
        sift_score.append('nan')


#concatenate scores as features
dna_feature = pd.DataFrame(
    {'cadd': cadd_score,
     'fathmm': fathmm_score,
     'vest': vest_score,
     'proven': proven_score,
     'sift': sift_score
    })

#dna_feature.to_csv('data/feature/dna_feature.csv')





