#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:24:11 2021

@author: rayin
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import random

from collections import Counter
from pprint import pprint

os.chdir("/Users/rayin/Google Drive/Harvard/5_data/UDN/work")

case_gene_update = pd.read_csv("data/processed/variant_clean.csv", index_col=0)
aa_variant = list(case_gene_update['\\12_Candidate variants\\09 Protein\\'])
#pd.DataFrame(aa_variant).to_csv('aa_variant.csv')

#aa_variant_update = pd.read_csv("data/processed/aa_variant_update.csv", index_col=0)
#aa_variant_update = list(aa_variant_update['\\12_Candidate variants\\09 Protein\\'])

amino_acid = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
              'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'TER': 'X'}

aa_3 = []
aa_1 = []
for i in amino_acid.keys():
    aa_3.append(i)
    aa_1.append(amino_acid[i])

for i in range(0, len(aa_variant)):
    for j in range(len(aa_3)):
        if isinstance(aa_variant[i], float):
            break
        aa_variant[i] = str(aa_variant[i].upper())
        if aa_3[j] in aa_variant[i]:
            aa_variant[i] = aa_variant[i].replace(aa_3[j], aa_1[j])
            
#extracting aa properties from aaindex
#https://www.genome.jp/aaindex/
aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] 

#RADA880108           
polarity = [-0.06, -0.84, -0.48, -0.80, 1.36, -0.73, -0.77, -0.41, 0.49, 1.31, 1.21, -1.18, 1.27, 1.27, 0.0, -0.50, -0.27, 0.88, 0.33, 1.09]
aa_polarity = pd.concat([pd.Series(aa), pd.Series(polarity)], axis=1)
aa_polarity = aa_polarity.rename(columns={0:'amino_acid', 1: 'polarity_value'})

#KLEP840101
net_charge = [0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
aa_net_charge = pd.concat([pd.Series(aa), pd.Series(net_charge)], axis=1)
aa_net_charge = aa_net_charge.rename(columns={0:'amino_acid', 1: 'net_charge_value'})

#CIDH920103
hydrophobicity = [0.36, -0.52, -0.90, -1.09, 0.70, -1.05, -0.83, -0.82, 0.16, 2.17, 1.18, -0.56, 1.21, 1.01, -0.06, -0.60, -1.20, 1.31, 1.05, 1.21]
aa_hydrophobicity = pd.concat([pd.Series(aa), pd.Series(hydrophobicity)], axis=1)
aa_hydrophobicity = aa_hydrophobicity.rename(columns={0:'amino_acid', 1: 'hydrophobicity_value'})

#FAUJ880103 -- Normalized van der Waals volume
normalized_vdw = [1.00, 6.13, 2.95, 2.78, 2.43, 3.95, 3.78, 0.00, 4.66, 4.00, 4.00, 4.77, 4.43, 5.89, 2.72, 1.60, 2.60, 8.08, 6.47, 3.00]
aa_normalized_vdw = pd.concat([pd.Series(aa), pd.Series(normalized_vdw)], axis=1)
aa_normalized_vdw = aa_normalized_vdw.rename(columns={0:'amino_acid', 1: 'normalized_vdw_value'})

#CHAM820101
polarizability = [0.046, 0.291, 0.134, 0.105, 0.128, 0.180, 0.151, 0.000, 0.230, 0.186, 0.186, 0.219, 0.221, 0.290, 0.131, 0.062, 0.108, 0.409, 0.298, 0.140]
aa_polarizability = pd.concat([pd.Series(aa), pd.Series(polarizability)], axis=1)
aa_polarizability = aa_polarizability.rename(columns={0:'amino_acid', 1: 'polarizability_value'})

#JOND750102
pK_COOH = [2.34, 1.18, 2.02, 2.01, 1.65, 2.17, 2.19, 2.34, 1.82, 2.36, 2.36, 2.18, 2.28, 1.83, 1.99, 2.21, 2.10, 2.38, 2.20, 2.32]
aa_pK_COOH = pd.concat([pd.Series(aa), pd.Series(pK_COOH)], axis=1)
aa_pK_COOH = aa_pK_COOH.rename(columns={0:'amino_acid', 1: 'pK_COOH_value'})

#FASG760104
pK_NH2 = [9.69, 8.99, 8.80, 9.60, 8.35, 9.13, 9.67, 9.78, 9.17, 9.68, 9.60, 9.18, 9.21, 9.18, 10.64, 9.21, 9.10, 9.44, 9.11, 9.62]
aa_pK_NH2 = pd.concat([pd.Series(aa), pd.Series(pK_NH2)], axis=1)
aa_pK_NH2 = aa_pK_NH2.rename(columns={0:'amino_acid', 1: 'pK_NH2_value'})

#ROBB790101  Hydration free energy
hydration = [-1.0, 0.3, -0.7, -1.2, 2.1, -0.1, -0.7, 0.3, 1.1, 4.0, 2.0, -0.9, 1.8, 2.8, 0.4, -1.2, -0.5, 3.0, 2.1, 1.4]
aa_hydration = pd.concat([pd.Series(aa), pd.Series(hydration)], axis=1)
aa_hydration = aa_hydration.rename(columns={0:'amino_acid', 1: 'hydration_value'})

#FASG760101
molecular_weight = [89.09, 174.20, 132.12, 133.10, 121.15, 146.15, 147.13, 75.07, 155.16, 131.17, 131.17, 146.19, 149.21, 165.19, 
                    115.13, 105.09, 119.12, 204.24, 181.19, 117.15]
aa_molecular_weight = pd.concat([pd.Series(aa), pd.Series(molecular_weight)], axis=1)
aa_molecular_weight = aa_molecular_weight.rename(columns={0:'amino_acid', 1: 'molecular_weight_value'})

#FASG760103
optical_rotation = [1.80, 12.50, -5.60, 5.05, -16.50, 6.30, 12.00, 0.00, -38.50, 12.40, -11.00, 14.60, -10.00, -34.50, -86.20,
                    -7.50, -28.00, -33.70, -10.00, 5.63]
aa_optical_rotation = pd.concat([pd.Series(aa), pd.Series(optical_rotation)], axis=1)
aa_optical_rotation = aa_optical_rotation.rename(columns={0:'amino_acid', 1: 'optical_rotation_value'})

#secondary structure #LEVJ860101
#https://pybiomed.readthedocs.io/en/latest/_modules/CTD.html#CalculateCompositionSolventAccessibility
#SecondaryStr = {'1': 'EALMQKRH', '2': 'VIYCWFT', '3': 'GNPSD'}
# '1'stand for Helix; '2'stand for Strand, '3' stand for coil
secondary_structure = [1, 1, 3, 3, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 3, 3, 2, 2, 2, 2]
aa_secondary_structure = pd.concat([pd.Series(aa), pd.Series(secondary_structure)], axis=1)
aa_secondary_structure = aa_secondary_structure.rename(columns={0:'amino_acid', 1: 'secondary_structure_value'})

#_SolventAccessibility = {'-1': 'ALFCGIVW', '1': 'RKQEND', '0': 'MPSTHY'}
# '-1'stand for Buried; '1'stand for Exposed, '0' stand for Intermediate
solvent_accessibility = [-1, 1, 1, 1, -1, 1, 1, -1, 0, -1, -1, 1, 0, -1, 0, 0, 0, -1, 0, -1]
aa_solvent_accessibility = pd.concat([pd.Series(aa), pd.Series(solvent_accessibility)], axis=1)
aa_solvent_accessibility = aa_solvent_accessibility.rename(columns={0:'amino_acid', 1: 'solvent_accessibility_value'})

############################################################################################################################################
#CHAM820102 Free energy of solution in water
free_energy_solution = [-0.368, -1.03, 0.0, 2.06, 4.53, 0.731, 1.77, -0.525, 0.0, 0.791, 1.07, 0.0, 0.656, 1.06, -2.24, -0.524, 0.0, 1.60, 4.91, 0.401]
aa_free_energy_solution = pd.concat([pd.Series(aa), pd.Series(free_energy_solution)], axis=1)
aa_free_energy_solution = aa_free_energy_solution.rename(columns={0:'amino_acid', 1: 'free_energy_solution_value'})
 
#FAUJ880109  Number of hydrogen bond donors
number_of_hydrogen_bond = [0, 4, 2, 1, 0, 2, 1, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 0]
aa_number_of_hydrogen_bond = pd.concat([pd.Series(aa), pd.Series(number_of_hydrogen_bond)], axis=1)
aa_number_of_hydrogen_bond = aa_number_of_hydrogen_bond.rename(columns={0:'amino_acid', 1: 'number_of_hydrogen_bond_value'})

#PONJ960101 Average volumes of residues
volumes_of_residues = [91.5, 196.1, 138.3, 135.2, 114.4, 156.4, 154.6, 67.5, 163.2, 162.6, 163.4, 162.5, 165.9, 198.8, 123.4, 102.0, 126.0, 209.8, 237.2, 138.4]
aa_volumes_of_residues = pd.concat([pd.Series(aa), pd.Series(volumes_of_residues)], axis=1)
aa_volumes_of_residues = aa_volumes_of_residues.rename(columns={0:'amino_acid', 1: 'volumes_of_residues_value'})

#JANJ790102
transfer_free_energy = [0.3, -1.4, -0.5, -0.6, 0.9, -0.7, -0.7, 0.3, -0.1, 0.7, 0.5, -1.8, 0.4, 0.5, -0.3, -0.1, -0.2, 0.3, -0.4, 0.6]
aa_transfer_free_energy = pd.concat([pd.Series(aa), pd.Series(transfer_free_energy)], axis=1)
aa_transfer_free_energy = aa_transfer_free_energy.rename(columns={0:'amino_acid', 1: 'transfer_free_energy_value'})

#WARP780101 amino acid side-chain interactions in 21 proteins
side_chain_interaction = [10.04, 6.18, 5.63, 5.76, 8.89, 5.41, 5.37, 7.99, 7.49, 8.7, 8.79, 4.40, 9.15, 7.98, 7.79, 7.08, 7.00, 8.07, 6.90, 8.88]
aa_side_chain_interaction = pd.concat([pd.Series(aa), pd.Series(side_chain_interaction)], axis=1)
aa_side_chain_interaction = aa_side_chain_interaction.rename(columns={0:'amino_acid', 1: 'side_chain_interaction_value'})

#KARS160101 
number_of_vertices = [2.00, 8.00, 5.00, 5.00, 3.00, 6.00, 6.00, 1.00, 7.00, 5.00, 5.00, 6.00, 5.00, 8.00, 4.00, 3.00, 4.00, 11.00, 9.00, 4.00]
aa_number_of_vertices = pd.concat([pd.Series(aa), pd.Series(number_of_vertices)], axis=1)
aa_number_of_vertices = aa_number_of_vertices.rename(columns={0:'amino_acid', 1: 'number_of_vertices_value'})

#KARS160102 Number of edges (size of the graph)
number_of_edges = [1.00, 7.00, 4.00, 4.00, 2.00, 5.00, 5.00, 0.00, 6.00, 4.00, 4.00, 5.00, 4.00, 8.00, 4.00, 2.00, 3.00, 12.00, 9.00, 3.00]
aa_number_of_edges = pd.concat([pd.Series(aa), pd.Series(number_of_edges)], axis=1)
aa_number_of_edges = aa_number_of_edges.rename(columns={0:'amino_acid', 1: 'number_of_edges_value'})

#KARS160105
eccentricity = [1.00, 8.120, 5.00, 5.17, 2.33, 5.860, 6.00, 0.00, 6.71, 3.25, 5.00, 7.00, 5.40, 7.00, 4.00, 1.670, 3.250, 11.10, 8.88, 3.25]
aa_eccentricity = pd.concat([pd.Series(aa), pd.Series(eccentricity)], axis=1)
aa_eccentricity = aa_eccentricity.rename(columns={0:'amino_acid', 1: 'eccentricity_value'})

#KARS160107
diameter = [1.00, 12.00, 6.00, 6.00, 3.00, 8.00, 8.00, 0.00, 9.00, 6.00, 6.00, 9.00, 7.00, 11.000, 4.000, 3.00, 4.00, 14.000, 13.000, 4.00]
aa_diameter = pd.concat([pd.Series(aa), pd.Series(diameter)], axis=1)
aa_diameter = aa_diameter.rename(columns={0:'amino_acid', 1: 'diameter_value'})

#KARS160117  Total weighted atomic number of the graph
atomic_number = [12.00, 45.00, 33.007, 34.00, 28.00, 39.00, 40.00, 7.00, 47.00, 30.0, 30.00, 37.00, 40.00, 48.00, 24.00, 22.00, 27.00, 68.00, 56.00, 24.00]
aa_atomic_number = pd.concat([pd.Series(aa), pd.Series(atomic_number)], axis=1)
aa_atomic_number = aa_atomic_number.rename(columns={0:'amino_acid', 1: 'atomic_number_value'})


def replace_uncertain_aa(amino_acid):
  """
  Randomly selects replacements for all uncertain amino acids.
  Expects and returns a string.
  """
  replacements = {'B': 'DN',
                  'J': 'IL',
                  'Z': 'EQ',
                  'X': 'ACDEFGHIKLMNPQRSTVWY'}
  replace_aa = amino_acid.replace(amino_acid, random.choice(replacements[amino_acid]))
  return replace_aa


def aa_change(df):
    aa_1 = []
    aa_2 = []
    for i in range(0, len(df)):
        if isinstance(df[i], float):
            aa_1.append('')
            aa_2.append('')
            continue
        elif isinstance(df[i], str):
            aa_front = re.findall('P.(\D)', df[i])
            aa_front = ','.join(aa_front)
            aa_back = re.findall('[0-9](\D.*)', df[i])
            aa_back = ','.join(aa_back)
            if aa_front == 'X':
                aa_front = replace_uncertain_aa(aa_front)
            elif aa_front == 'FS':
                aa_front = aa_front.replace(aa_front, '')
            elif aa_front == 'DEL':
                aa_front = aa_front.replace(aa_front, '')
                
            if aa_back == 'X':
                aa_back = replace_uncertain_aa(aa_back)
            elif aa_back == 'FS':
                aa_back = aa_back.replace(aa_back, '')
            elif aa_back == 'DEL':
                aa_back = aa_back.replace(aa_back, '')
            aa_1.append(aa_front)
            aa_2.append(aa_back)
        else:
            print('error')
            print(i)
            
    aa_change = pd.DataFrame({'old aa': aa_1,'new aa': aa_2})
    return aa_change

aa_change = aa_change(aa_variant)

#calculate the numeical different between two animo acids based on physiochemical properties
def property_transform(df, aa_property):
    df_column = list(df.columns.values)
    property_column = list(aa_property.columns.values)
    property_value = []
    for i in range(0, len(df)):
        flag_1 = 0
        flag_2 = 0
        for j in range(0, len(aa_property)):
            if df[df_column[0]].iloc[i] == aa_property[property_column[0]].iloc[j]:
                flag_1 = 1
                value_1 = aa_property[property_column[1]].iloc[j]
            if df[df_column[1]].iloc[i] == aa_property[property_column[0]].iloc[j]:
                value_2 = aa_property[property_column[1]].iloc[j]
                flag_2 = 1
            if flag_1 == 1 and flag_2 == 1:
                value = round(value_1 - value_2, 3)
            elif flag_1 == 1 and flag_2 == 0:
                value = -value_1
            elif flag_1 == 0 and flag_2 == 1:
                value = value_2
            elif flag_1 == 0 and flag_2 == 0:
                value = ''
        property_value.append(value)
    property_value = pd.DataFrame(property_value, columns=[property_column[1]+'_diff'])
    return property_value


polarity_diff = property_transform(aa_change, aa_polarity)
net_charge_diff = property_transform(aa_change, aa_net_charge)
hydrophobicity_diff = property_transform(aa_change, aa_hydrophobicity)
normalized_vdw_diff = property_transform(aa_change, aa_normalized_vdw)
polarizability_diff = property_transform(aa_change, aa_polarizability)
pK_COOH_diff = property_transform(aa_change, aa_pK_COOH)
pK_NH2_diff = property_transform(aa_change, aa_pK_NH2)
hydration_diff = property_transform(aa_change, aa_hydration)
molecular_weight_diff = property_transform(aa_change, aa_molecular_weight)
optical_rotation_diff = property_transform(aa_change, aa_optical_rotation)
secondary_structure_diff = property_transform(aa_change, aa_secondary_structure)


free_energy_solution_diff = property_transform(aa_change, aa_free_energy_solution)
number_of_hydrogen_bond_diff = property_transform(aa_change, aa_number_of_hydrogen_bond)
volumes_of_residues_diff = property_transform(aa_change, aa_volumes_of_residues)
transfer_free_energy_diff = property_transform(aa_change, aa_transfer_free_energy)
side_chain_interaction_diff = property_transform(aa_change, aa_side_chain_interaction)
number_of_vertices_diff = property_transform(aa_change, aa_number_of_vertices)
number_of_edges_diff = property_transform(aa_change, aa_number_of_edges)
eccentricity_diff = property_transform(aa_change, aa_eccentricity)
diameter_diff = property_transform(aa_change, aa_diameter)
atomic_number_diff = property_transform(aa_change, aa_atomic_number)

aa_feature = pd.concat([polarity_diff, net_charge_diff, hydrophobicity_diff, normalized_vdw_diff, polarizability_diff, pK_COOH_diff,
                        pK_NH2_diff, hydration_diff, molecular_weight_diff, optical_rotation_diff, secondary_structure_diff,
                        free_energy_solution_diff, number_of_hydrogen_bond_diff, volumes_of_residues_diff,
                        transfer_free_energy_diff, side_chain_interaction_diff, number_of_vertices_diff, number_of_edges_diff,
                        eccentricity_diff, diameter_diff, atomic_number_diff], axis=1)
aa_feature.to_csv('data/feature/aa_feature.csv')


gene_feature = pd.read_csv("data/feature/gene_feature.csv", index_col=0)
dna_feature = pd.read_csv("data/feature/dna_feature.csv", index_col=0)
feature = pd.concat([gene_feature, aa_feature, dna_feature], axis=1)
#feature.to_csv('data/feature/feature.csv')


