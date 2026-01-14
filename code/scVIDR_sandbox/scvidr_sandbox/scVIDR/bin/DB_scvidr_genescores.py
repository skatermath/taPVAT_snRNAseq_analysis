# Access to code
print('Beging loading modules')
import os
import sys
from datetime import datetime

# Match training script's sys.path setup

sys.path.insert(0, "/mnt/scratch/bowmand8/scVIDR/scvidr_sandbox/scVIDR")

from vidr.vidr import *


import argparse
import logging
logging.basicConfig(level=logging.INFO)

import scanpy as sc


from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import torch
from scipy import spatial, linalg

print('Modules loaded')





# 24W Male

# '../models/VIDR_August_13_2025_1449_Adipocytes_Brown_24W_M_100'

#### Adipocytes_Brown 24 Weeks Male
# CUDA_VISIBLE_DEVICES=1 python ./DB_scvidr_genescores.py \
# --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad' \
# --model_path '../models/VIDR_August_13_2025_1449_Adipocytes_Brown_24W_M_100' \
# --save_path '../predictions/gene_scores/' \
# --celltype_to_predict 'Adipocytes_Brown' \
# --condition_to_predict 'HF' \
# --celltype_column 'celltype' \
# --condition_column_name 'diet' \
# --timeframe_to_consider '24W' \
# --sex_to_consider 'M' 



# 8W Male

# ../models/VIDR_August_13_2025_1531_Adipocytes_Brown_8W_M_100'


#### Adipocytes_Brown 8 Weeks Male
# CUDA_VISIBLE_DEVICES=1 python ./DB_scvidr_genescores.py \
# --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad' \
# --model_path '../models/VIDR_August_13_2025_1531_Adipocytes_Brown_8W_M_100' \
# --save_path '../predictions/gene_scores/' \
# --celltype_to_predict 'Adipocytes_Brown' \
# --condition_to_predict 'HF' \
# --celltype_column 'celltype' \
# --condition_column_name 'diet' \
# --timeframe_to_consider '8W' \
# --sex_to_consider 'M' 



#24W Female

# '../models/VIDR_August_13_2025_1516_Adipocytes_Brown_24W_F_100'


#### Adipocytes_Brown 24 Weeks Female
# CUDA_VISIBLE_DEVICES=1 python ./DB_scvidr_genescores.py \
# --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad' \
# --model_path '../models/VIDR_August_13_2025_1516_Adipocytes_Brown_24W_F_100' \
# --save_path '../predictions/gene_scores/' \
# --celltype_to_predict 'Adipocytes_Brown' \
# --condition_to_predict 'HF' \
# --celltype_column 'celltype' \
# --condition_column_name 'diet' \
# --timeframe_to_consider '24W' \
# --sex_to_consider 'F' 




#8W Female

# '../models/VIDR_August_13_2025_1529_Adipocytes_Brown_8W_F_100'

# ### Adipocytes_Brown 8 Weeks Female
# CUDA_VISIBLE_DEVICES=1 python ./DB_scvidr_genescores.py \
# --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad' \
# --model_path '../models/VIDR_August_13_2025_1529_Adipocytes_Brown_8W_F_100' \
# --save_path '../predictions/gene_scores/' \
# --celltype_to_predict 'Adipocytes_Brown' \
# --condition_to_predict 'HF' \
# --celltype_column 'celltype' \
# --condition_column_name 'diet' \
# --timeframe_to_consider '8W' \
# --sex_to_consider 'F' 




print('Starting argument parsing...')

parser = argparse.ArgumentParser(description='Interpret scVIDR predictions using ridge regression.')

parser.add_argument('--h5ad_file_path', required=True, help='The data file containing the raw reads in h5ad format')
parser.add_argument('--model_path', required=True, help='Path to the trained model')
parser.add_argument('--save_path', default='../predictions/gene_scores/', required=True)

parser.add_argument('--celltype_to_predict', required=True, help='Name of the cell type to predict')
parser.add_argument('--condition_to_predict', required=True, help='Name of the condition to predict')
parser.add_argument('--celltype_column', default='celltype')
parser.add_argument('--condition_column_name', required=True, help='Name of the column containing condition information')
parser.add_argument('--timeframe_to_consider', required=True, help='Name of the timeframe to consider (8W or 24W)')
parser.add_argument('--sex_to_consider', required=True, default='Both', help='Sex to consider (M,F,Both)')

parser.add_argument('--training_size', default=100000, type=int)
args = parser.parse_args()

# loading CLI arguments
DATA_PATH = args.h5ad_file_path
MODEL_PATH = args.model_path
SAVE_PATH = args.save_path

CELLTYPE_TO_PREDICT = args.celltype_to_predict
CONDITION_TO_PREDICT = args.condition_to_predict
CELLTYPE_COLUMN = args.celltype_column
CONDITION_COLUMN = args.condition_column_name
TIMEFRAME_TO_CONSIDER = args.timeframe_to_consider
SEX_TO_CONSIDER = args.sex_to_consider


NUM_SAMPLES = args.training_size

logging.info(f'Loading data file: {DATA_PATH}\n\n')
# ---------------- Load and Filter Data ---------------- #
adata = sc.read_h5ad(DATA_PATH)
adata.obs['dose'] = adata.obs[CONDITION_COLUMN].astype(str)  # reserve "dose" as internal


celltypes = adata.obs[CELLTYPE_COLUMN].unique().tolist()

CELLTYPES_OF_INTEREST = []
for cell_type in celltypes:                  
    if cell_type != CELLTYPE_TO_PREDICT:
        CELLTYPES_OF_INTEREST.append(cell_type)

available_conditions = adata.obs[CONDITION_COLUMN].unique()

# normalize and preprocess data
##########################
logging.info(f'\n\nNormalizing and preparing data...')
adata = adata[adata.obs['time'] == TIMEFRAME_TO_CONSIDER]
if SEX_TO_CONSIDER != 'Both':
    adata = adata[adata.obs['sex'] == SEX_TO_CONSIDER]
print(f"Sex of animals: {adata.obs['sex'].unique()}")

celltype_counts = adata.obs['celltype'].value_counts()

adata = normalize_data(adata)
logging.info('---- DONE')

available_cell_types = adata.obs[CELLTYPE_COLUMN].unique()
for cell_type in available_cell_types:
    logging.info(f'--- {cell_type}')

         

print(f'Adata shape: {adata.shape}')
train_adata, test_adata = prepare_data(
    adata=adata, 
    celltype_column_name=CELLTYPE_COLUMN, 
    condition_column_name=CONDITION_COLUMN, 
    celltype_to_predict=CELLTYPE_TO_PREDICT, 
    condition_to_predict=CONDITION_TO_PREDICT)



print(f'Train_adata shape after prepare_data function: {train_adata.shape}')


logging.info(f'Loading model: {MODEL_PATH}') 
model = VIDR(train_adata, linear_decoder = False)
model = model.load(MODEL_PATH, train_adata)




#print('STATEMENT RIGHT BEFORE DEFINING MODEL_PREDICT IN DB_SCVIDR_GENESCORES.PY')

def model_predict(model, control_dose, treated_dose, test_celltype, dose_column_type, regression=True):
    pred, delta, *other = model.predict(
        ctrl_key=control_dose,
        treat_key=treated_dose,
        cell_type_to_predict=test_celltype,
        regression=regression
    )

    print('this is inside the definition!!!!!')
    
    print(pred)
    reg = other[0] if other else None

    pred.obs[CONDITION_COLUMN] = treated_dose
    pred.obs[CONDITION_COLUMN] = pred.obs[CONDITION_COLUMN].astype(dose_column_type)
    pred.obs['diet'] = f'{treated_dose}'

    return {treated_dose: pred}, delta, reg



dose_column_type = adata.obs[CONDITION_COLUMN].dtype

#print('STATEMENT RIGHT BEFORE ACTUALLY RUNNING THE MODEL_PREDICT FUNCTION IN DB_SCVIDR_GENESCORES.PY')
# predict and scVIDR cell data
pred, delta, reg = model_predict(
    model=model,
    control_dose='Control', 
    treated_dose=CONDITION_TO_PREDICT, 
    test_celltype=CELLTYPE_TO_PREDICT, 
    dose_column_type=dose_column_type,
    regression=True
)

print(pred)
print(delta)
print(reg)

#Generating input dataset
latent = model.get_latent_representation()

print(f' Latent shape: {latent.shape}')
mins = np.min(latent, axis = 0)

print(f'Minimums: {mins}.')
print(f'Minimums shape: {mins.shape}')

maxes = np.max(latent, axis = 0)


print(f'Maximums: {maxes}.')
print(f'Maximums shape: {maxes.shape}')


rand_samp = []
for (mn,mx) in zip(mins, maxes):
    rand_samp += [np.random.uniform(mn,mx,  size = (NUM_SAMPLES))]
rand_samp = np.array(rand_samp).T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen_samp = model.module.generative(torch.from_numpy(rand_samp).float())["px"].cpu().detach().numpy()

#Weighting sample based on distance from delta
weights = []
for i in range(rand_samp.shape[0]):
    weights += [1/spatial.distance.cosine(rand_samp[i, :], delta)]

# --- NaN filtering before regression ---
X = np.array(rand_samp)
y = np.array(gen_samp)

print(f"Rows before NaN filtering: {X.shape[0]}")
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
X_clean = X[mask]
y_clean = y[mask]
weights_clean = np.array(weights)[mask]

print(f"Rows after NaN filtering: {X_clean.shape[0]}")
print(f"Rows dropped: {X.shape[0] - X_clean.shape[0]}")


if X_clean.shape[0] == 0:
    raise ValueError("No valid rows remaining after NaN filtering — cannot fit regression.")

# Fitting Ridge Model
reg = Ridge()
reg.fit(X_clean, y_clean, sample_weight=weights_clean)

#Generating Test Set For Evaluating Model
test_samp = []
for (mn,mx) in zip(mins, maxes):
    test_samp += [np.random.uniform(mn,mx,  size = (20000))]
test_samp = np.array(test_samp).T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_gen_samp = model.module.generative(torch.from_numpy(test_samp).float())["px"].cpu().detach().numpy()

#Ridge Regression Score
eval_score = reg.score(test_samp,test_gen_samp)
logging.info(f"---- Ridge Regression Accuracy on Test Set: {eval_score}")

#Generating gene scores
gene_weights = reg.coef_
gene_norms = linalg.norm(reg.coef_, axis = 0)
gene_weights_norm = gene_weights / gene_norms

#Creating Pandas DataFrame
gene_scores = np.dot(gene_weights_norm, delta[:, np.newaxis]).squeeze()
gene_names = model.adata.var_names

gene_df = pd.DataFrame({"Gene":gene_names, "Score":gene_scores})

print('Gene dataframe created.')

# TODO: save pred_dict
#Check if output directory exists and create one if it doesn't
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    
from datetime import datetime

timestamp = datetime.now().strftime("%B%d_%H_%M")

#Save output
gene_df.to_csv(f"{SAVE_PATH}/{timestamp}_gene_scores_{CELLTYPE_TO_PREDICT}_{TIMEFRAME_TO_CONSIDER}_{SEX_TO_CONSIDER}.csv")
logging.info(f"---- Saved GENE_SCORES: {SAVE_PATH}/{timestamp}_gene_scores_{CELLTYPE_TO_PREDICT}_{TIMEFRAME_TO_CONSIDER}_{SEX_TO_CONSIDER}.csv")
