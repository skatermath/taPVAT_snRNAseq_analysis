#Create Access to my code
import os
import sys

from datetime import datetime
# sys.path.append(os.path.abspath('vidr/'))
# sys.path.append(os.path.abspath('data/'))

path_to_add = os.path.abspath('/scVIDR')
sys.path.insert(0, path_to_add)
print(f"Added to sys.path: {path_to_add}")

print("Full sys.path:")
for p in sys.path:
    print("  ", p)
    
import argparse
import scanpy as sc
import logging
logging.basicConfig(level=logging.INFO)


from vidr import VIDR, normalize_data, prepare_data



#### Adipocytes (Broad)

# python DB_scvidr_train.py \
#   --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad'\
#   --save_path '../models/' \
#   --condition_to_predict 'HF' \
#   --celltype_column_name 'celltype_broad' \
#   --celltype_to_predict 'Adipocytes' \
#   --condition_column_name 'diet' \
#   --timeframe_to_consider '24W' \
#   --sex_to_consider 'Both'



#### Adipocytes_Brown

# CUDA_VISIBLE_DEVICES=1 python DB_scvidr_train.py \
#   --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad'\
#   --save_path '../models/' \
#   --condition_to_predict 'HF' \
#   --celltype_column_name 'celltype' \
#   --celltype_to_predict 'Adipocytes_Brown' \
#   --condition_column_name 'diet' \
#   --timeframe_to_consider '8W' \
#   --sex_to_consider 'M'\
#   --max_epochs=100





#### ECs_Cap
# python DB_scvidr_train.py \
#   --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad'\
#   --save_path '../models/' \
#   --condition_to_predict 'HF' \
#   --celltype_column_name 'celltype' \
#   --celltype_to_predict 'ECs_Cap' \
#   --condition_column_name 'diet' \
#   --timeframe_to_consider '24W' \
#   --sex_to_consider 'Both'






parser = argparse.ArgumentParser(description='Train a VAE model applicable to scGen and scVIDR using a h5ad input dataset')
parser.add_argument('--h5ad_file_path', required=True, help='The data file containing the raw reads in h5ad format')
parser.add_argument('--save_path', required=True, help='Path to the directory where the trained model will be saved')
parser.add_argument('--celltype_to_predict', required=True, help='Name of the cell type to predict')
parser.add_argument('--condition_to_predict', required=True, help='Name of the condition to predict')
parser.add_argument('--celltype_column_name', required=True, help='Name of the column containing cell type information')
parser.add_argument('--condition_column_name', required=True, help='Name of the column containing condition information')
parser.add_argument('--timeframe_to_consider', required=True, help='Name of the timeframe to consider (8W or 24W)')
parser.add_argument('--sex_to_consider', required=True, default='Both', help='Sex to consider (M,F,Both)')
parser.add_argument('--max_epochs', type=int, required=True, default=100, help='Number of training epochs')

script_args = parser.parse_args()

# loading CLI arguments
DATA_PATH = script_args.h5ad_file_path
MODEL_OUTPUT_DIR = script_args.save_path
CELLTYPE_TO_PREDICT = script_args.celltype_to_predict
CONDITION_TO_PREDICT = script_args.condition_to_predict
CELLTYPE_COLUMN = script_args.celltype_column_name
CONDITION_COLUMN = script_args.condition_column_name
TIMEFRAME_TO_CONSIDER = script_args.timeframe_to_consider
SEX_TO_CONSIDER = script_args.sex_to_consider
NUMBER_OF_EPOCHS = script_args.max_epochs


logging.info(f'Loading data file: {DATA_PATH}\n\n')
adata = sc.read_h5ad(DATA_PATH)
adata = adata[adata.obs['time'] == TIMEFRAME_TO_CONSIDER]
if SEX_TO_CONSIDER != 'Both':
    adata = adata[adata.obs['sex'] == SEX_TO_CONSIDER]
print(f"Sex of animals: {adata.obs['sex'].unique()}")

# Compute counts per cell type
celltype_counts = adata.obs['celltype'].value_counts()

# # Filter to only those with at least 500 cells
# valid_celltypes = celltype_counts[celltype_counts >= 500].index
# print(f"Retained {len(valid_celltypes)} cell types with ≥500 cells:")
# print(valid_celltypes.tolist())

# # Subset the AnnData object
# adata = adata[adata.obs['celltype'].isin(valid_celltypes)]


# Cell type checks
#########################
logging.info(f'Cell types available in the dataset: ')
available_cell_types = adata.obs[CELLTYPE_COLUMN].unique()
for cell_type in available_cell_types:
    logging.info(f'--- {cell_type}')

# Check if the specified celltype_to_predict exists
if CELLTYPE_TO_PREDICT not in available_cell_types:
    raise ValueError(f'Cell type to predict "{CELLTYPE_TO_PREDICT}" not found in available cell types: {available_cell_types}')
else:
    logging.info(f'Cell type to predict: {CELLTYPE_TO_PREDICT} EXISTS')

# Condition checks
#########################
logging.info(f'Conditions available in the dataset: ')
available_conditions = adata.obs[CONDITION_COLUMN].unique()
for condition in available_conditions:
    logging.info(f'--- {condition}')

# Check if the specified condition_to_predict exists
if CONDITION_TO_PREDICT not in available_conditions:
    raise ValueError(f'Condition to predict "{CONDITION_TO_PREDICT}" not found in available conditions: {available_conditions}')
else:
    logging.info(f'Condition to predict: {CONDITION_TO_PREDICT} EXISTS')

logging.info('---- VALIDATION DONE')

# normalize and preprocess data
##########################

logging.info(f'\n\nNormalizing and preparing data...')

adata_filtered = normalize_data(adata)


logging.info('---- DONE')

logging.info(f'\n\nTraining model...')



print(adata_filtered.shape)
  # Using 'diet' as the batch column so the model learns to correct for the control vs. HF. This will be used to find  perturbation vector (in theory).)
train_adata, test_adata = prepare_data(
    adata=adata_filtered, 
    celltype_column_name=CELLTYPE_COLUMN, 
    condition_column_name=CONDITION_COLUMN, 
    celltype_to_predict=CELLTYPE_TO_PREDICT, 
    condition_to_predict=CONDITION_TO_PREDICT)


print(f'Train_adata shape after prepare_data function: {train_adata.shape}')


model = VIDR(train_adata, linear_decoder = False)


model.train(
    max_epochs=NUMBER_OF_EPOCHS,
    batch_size=128,
    early_stopping=True,
    early_stopping_patience=25
)
logging.info('---- DONE')


number_of_epochs_string = str(NUMBER_OF_EPOCHS)

timestamp = datetime.now().strftime("%B_%d_%Y_%H%M")
save_path = os.path.join(MODEL_OUTPUT_DIR, f"VIDR_{timestamp}_{CELLTYPE_TO_PREDICT}_{TIMEFRAME_TO_CONSIDER}_{SEX_TO_CONSIDER}_{number_of_epochs_string}")


logging.info(f"Saving model to: {save_path}")
model.save(save_path)
logging.info(f"Model saved to: {save_path}")
logging.info('---- DONE')
