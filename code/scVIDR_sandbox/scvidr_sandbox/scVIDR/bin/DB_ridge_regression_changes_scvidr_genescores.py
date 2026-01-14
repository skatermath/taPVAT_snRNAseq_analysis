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


import numpy as np
import logging
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

# ---- helpers ----
def cosine_weights(X, delta, scheme="inv_dist", eps=1e-3, temp=5.0):
    """
    Vectorized weights per row of X based on cosine w.r.t. delta.
    - 'inv_dist': w = 1 / max(1 - cos, eps)  (cap singularities)
    - 'exp':      w = exp(temp * cos)        (softer, always finite)
    Returns weights normalized to mean 1.
    """
    X = np.asarray(X, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    dnorm = np.linalg.norm(delta) + 1e-12
    xnorm = np.linalg.norm(X, axis=1) + 1e-12
    cos = (X @ delta) / (dnorm * xnorm)

    if scheme == "inv_dist":
        dist = 1.0 - cos
        w = 1.0 / np.maximum(dist, eps)
    elif scheme == "exp":
        w = np.exp(temp * cos)
    else:
        w = np.ones_like(cos)

    # Normalize to mean 1 so alpha has a sensible scale
    w *= (w.size / w.sum())
    return w

def minmax_scale(X, mins, maxes):
    mins = np.asarray(mins, dtype=np.float32)
    maxes = np.asarray(maxes, dtype=np.float32)
    return (X - mins) / (np.maximum(maxes - mins, 1e-12))

# ---- training data (your sampling & generator) ----
rand_samp = np.column_stack([
    np.random.uniform(mn, mx, size=NUM_SAMPLES) for mn, mx in zip(mins, maxes)
]).astype(np.float32)

gen_samp = (model.module.generative(
    torch.from_numpy(rand_samp).float()
)["px"].cpu().detach().numpy())

# ---- weights (stable) ----
weights = cosine_weights(rand_samp, delta, scheme="inv_dist", eps=1e-3)

# ---- NaN filter ----
X = rand_samp
y = gen_samp
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
X_clean = X[mask]
y_clean = y[mask]
w_clean = weights[mask]
if X_clean.shape[0] == 0:
    raise ValueError("No valid rows after NaN filtering.")

# ---- scale features BEFORE ridge (penalty is scale-dependent) ----
X_clean_s = minmax_scale(X_clean, mins, maxes)

# ---- choose lambda (alpha) by CV ----
alphas = np.logspace(-4, 4, 60).astype(np.float32)
reg = RidgeCV(alphas=alphas, store_cv_values=False)
reg.fit(X_clean_s, y_clean, sample_weight=w_clean)
best_alpha = reg.alpha_
logging.info(f"Chosen alpha (ridge λ): {best_alpha}")

# ---- test set ----
test_samp = np.column_stack([
    np.random.uniform(mn, mx, size=20000) for mn, mx in zip(mins, maxes)
]).astype(np.float32)

test_gen_samp = (model.module.generative(
    torch.from_numpy(test_samp).float()
)["px"].cpu().detach().numpy())

test_samp_s = minmax_scale(test_samp, mins, maxes)
y_pred = reg.predict(test_samp_s)

# ---- evaluation: unweighted and weighted R^2 (consistent with training weights) ----
r2_unweighted = r2_score(test_gen_samp, y_pred, multioutput="uniform_average")
w_test = cosine_weights(test_samp, delta, scheme="inv_dist", eps=1e-3)
r2_weighted = r2_score(test_gen_samp, y_pred, sample_weight=w_test, multioutput="uniform_average")

logging.info(f"R^2 (unweighted): {r2_unweighted:.4f}")
logging.info(f"R^2 (weighted):   {r2_weighted:.4f}")


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
