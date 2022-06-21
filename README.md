# VarPPUD
Repository with the code developed for the manuscript VarPPUD: predicting variant pathogenicity in the undiagnosed disease patients

## Authorizations
- To have access to the Undiagnosed Disease Network database, approval is required.
- Access to the data was done via PIC-SURE

## Repo organization and description
Code/: contains the files constructing the framework  
&emsp; - PIC-SURE/  
&emsp;&emsp;   HPDS_connection_manager.py: Build connection with PIC-SURE  
&emsp;&emsp;   utils.py: functions access data using PIC-SURE  
&emsp; - preprocess/  
&emsp;&emsp;    data_process.py: data preprocessing and cleaning based on inclusion criteria   
&emsp;&emsp;    feature_data_imputation.py: data imputation methods   
&emsp;&emsp;    data_generation.py: synthetic data generation using CTGAN through constraints  
&emsp; - feature/  
&emsp;&emsp;    feature_generation_gene.py: features generated based on genes  
&emsp;&emsp;    feature_generation_protein.py: feature generated based on protein variant    
&emsp;&emsp;    feature_generation_variant.py:  feature generated based on nucleotide variant   
&emsp; - model/  
&emsp;&emsp;    model.py: functions to implement prediction and plot ROC and PR curves   
&emsp;&emsp;    main.py: loading data and run the codes for prediction of variant pathogenicity   
&emsp;&emsp;    method_comparison.py: prediction results using other state-of-the-art methods   
&emsp; - analysis/  
&emsp;&emsp;    statistics.py: statistical analysis for inclusion patients data  
&emsp;&emsp;    visualization.py: figure visualizations   
             
Data/: Raw and intermediate data in the work   
&emsp;    - raw/: The raw data is avaialable upon request, accessing to the data was done via PIC-SURE  
&emsp;    - database/: databases used to generate features  
&emsp;    - feature/: numerical feature representations  

## Steps to run the codes
1. Clone the repository: git clone https://github.com/hms-dbmi/VarPPUD    
2. Change the the directory of all code files to the location where your data is accessed  
3. Run data_process.py to curate the inclusion patient information
4. Generate and concatenate different features through feature_generation_gene.py, feature_generation_protein.py and feature_generation_variant.py  
5. Generate synthetic data using data_generation.py for external validation of the model  
6. Run main.py to predict the variant pathogenicity of undiagnosed patients  
7. Run statistics.py and visualization.py for statistical analysis and visualization of input data and results  

## Publication
This code supports the analysis presented in: “VarPPUD: predicting variant pathogenicity in the undiagnosed disease patients”.


## License
Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
