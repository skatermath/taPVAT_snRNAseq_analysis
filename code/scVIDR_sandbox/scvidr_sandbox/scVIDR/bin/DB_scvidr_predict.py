import os
import sys
from datetime import datetime
import argparse
import scanpy as sc
import logging


# CUDA_VISIBLE_DEVICES=1 python DB_scvidr_predict.py --h5ad_file_path '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad' \
#     --model_path '../models/VIDR_August_07_2025_1152_Adipocytes_Brown_8W_Both' \
#     --output_path '../predictions/' \
#     --celltype_to_predict 'Adipocytes_Brown'\
#     --condition_to_predict 'HF' \
#     --celltype_column_name 'celltype' \
#     --condition_column_name 'diet' \






# Define your inputs here
# h5ad_file_path = '/mnt/research/sbhattacharya_lab/1_DerekBowman/scVIDR/DB_Updates/Full_taPVAT_combined_annotated_with_immune_fibro_ecs_all_genes_unnormalized.h5ad'

# 8W Models

# model_path = '../models/VIDR_August_07_2025_1207_Adipocytes_Brown_8W_F'
# model_path = '../models/VIDR_August_07_2025_1207_Adipocytes_Brown_8W_M'
#model_path = '../models/VIDR_August_07_2025_1152_Adipocytes_Brown_8W_Both'

# 24W Models

# model_path = '../models/VIDR_August_07_2025_1205_Adipocytes_Brown_24W_F'
#model_path = '../models/VIDR_August_07_2025_1204_Adipocytes_Brown_24W_M'
#model_path = '../models/VIDR_August_07_2025_1206_Adipocytes_Brown_24W_Both'

logging.basicConfig(level=logging.INFO)

# Setup project path
dev_path = os.path.abspath('/scVIDR')
sys.path.insert(0, dev_path)
print(f"Added to sys.path: {dev_path}")

from vidr import VIDR, normalize_data, prepare_data

# Argument parser
parser = argparse.ArgumentParser(description='Create cell predictions using a pretrained VAE model applicable to scGen and scVIDR using a h5ad input dataset')

parser.add_argument('--h5ad_file_path', required=True, help='The data file containing the raw reads in h5ad format')
parser.add_argument('--model_path', help='Path to the directory where the trained model was saved in the model training step')
parser.add_argument('--output_path', help='Path to the directory where the anndata will be output to in an h5ad format')
parser.add_argument('--celltype_to_predict', required=True, help='Name of the cell type to predict')
parser.add_argument('--condition_to_predict', required=True, help='Name of the condition to predict')
parser.add_argument('--celltype_column_name', required=True, help='Name of the column containing cell type information')
parser.add_argument('--condition_column_name', required=True, help='Name of the column containing condition information')
parser.add_argument('--control_dose', default='Control', help='Control condition (default: "Control")')
parser.add_argument('--treated_dose', default='HF', help='Treated condition (default: "HF")')


args = parser.parse_args()


# Load and preprocess data
logging.info(f'Loading data file: {args.h5ad_file_path}')
adata = sc.read_h5ad(args.h5ad_file_path)
adata = normalize_data(adata)

dose_column_type = adata.obs[args.condition_column_name].dtype

logging.info(f"Filtering data to exclude cell type '{args.celltype_to_predict}' under condition '{args.condition_to_predict}'")
train_adata, test_adata = prepare_data(
    adata=adata,
    celltype_column_name=args.celltype_column_name,
    condition_column_name=args.condition_column_name,
    celltype_to_predict=args.celltype_to_predict,
    condition_to_predict=args.condition_to_predict)

# Load model
logging.info(f'Loading model from {args.model_path}')
model = VIDR(train_adata, linear_decoder=False)
model = model.load(args.model_path, train_adata)

# Perform prediction
def model_predict(model, control_dose, treated_dose, test_celltype, dose_column_type):
    pred, delta, *other = model.predict(
        ctrl_key=control_dose,
        treat_key=treated_dose,
        cell_type_to_predict=test_celltype,
        regression=True
    )
    reg = other[0] if other else None

    pred.obs[args.condition_column_name] = treated_dose
    pred.obs[args.condition_column_name] = pred.obs[args.condition_column_name].astype(dose_column_type)
    pred.obs['dose'] = f'{treated_dose}'
   

    return {treated_dose: pred}, delta, reg

pred, delta, reg = model_predict(
    model=model,
    control_dose=args.control_dose,
    treated_dose=args.treated_dose,
    test_celltype=args.celltype_to_predict,
    dose_column_type=dose_column_type
)


import numpy as np

# Combine control, real treated, and predicted cells into one AnnData
# Make sure real_adata includes *both* control and treated cells
real_adata = adata[
    (adata.obs[celltype_column_name] == celltype_to_predict) &
    (adata.obs[condition_column_name].isin([control_dose, treated_dose]))
].copy()

pred_adata = pred[treated_dose]

# Confirm enough cells per group
print(real_adata.obs[condition_column_name].value_counts())

# Run differential expression
sc.tl.rank_genes_groups(real_adata, groupby=condition_column_name, method="wilcoxon")

# Extract top 100 DEGs for the treated group
top_genes_real = real_adata.uns["rank_genes_groups"]["names"][treated_dose][:100].tolist()



import numpy as np
import pandas as pd

# Ensure genes exist in both
shared_genes = [g for g in top_genes_real if g in pred_adata.var_names and g in real_adata.var_names]

def mean_expression(adata, genes):
    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.mean(X, axis=0)

mean_pred = mean_expression(pred_adata, shared_genes)
mean_real = mean_expression(real_adata, shared_genes)



import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

df = pd.DataFrame({
    "Predicted": mean_pred,
    "Observed": mean_real,
    "Gene": shared_genes
})

# Fit regression
slope, intercept, r_value, p_value, std_err = linregress(df["Predicted"], df["Observed"])
r_squared = r_value ** 2

# Plot
plt.figure(figsize=(6, 6))
sns.regplot(data=df, x="Predicted", y="Observed", scatter_kws={"s": 20})
plt.title(f"Top 100 Real DEGs (R² = {r_squared:.2f})")
plt.xlabel("Predicted Mean Expression")
plt.ylabel("Observed Mean Expression")

# Annotate top 10 genes
for i in range(min(10, len(shared_genes))):
    plt.text(df["Predicted"][i], df["Observed"][i], df["Gene"][i], fontsize=8)

plt.tight_layout()
os.makedirs("../DB_Figures/deg_plots", exist_ok=True)

plt.savefig(f"../DB_Figures/deg_plots/{celltype_to_predict}_deg_prediction.pdf")
plt.show()


























# Write predictions to disk
os.makedirs(args.output_path, exist_ok=True)
for dose, ad in pred.items():
    file_path = os.path.join(args.output_path, f"{dose}_PRED.h5ad")
    ad.write_h5ad(file_path)
    logging.info(f"Saved predicted output to: {file_path}")

logging.info('---- Prediction DONE')
