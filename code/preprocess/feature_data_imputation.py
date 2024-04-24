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




        
