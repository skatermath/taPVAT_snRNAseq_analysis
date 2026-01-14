#Import torch modules
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
#Import single cell modules
import scanpy as sc
import scvi
from scvi.data import setup_anndata
from scvi.dataloaders import AnnDataLoader

#Other important modules
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LinearRegression
from scipy import sparse
from scipy import stats





import scanpy as sc

def normalize_data(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    return adata[:,adata.var.highly_variable]

def prepare_data(
    adata, 
    celltype_column_name, 
    condition_column_name, 
    celltype_to_predict, 
    condition_to_predict):
    """
    Split data into training and testing sets:
    - Training: all cells EXCEPT the specified cell type under the specified condition
    - Testing: only the specified cell type under the specified condition
    """
    
    # Select test set: cells with specified cell type AND specified condition
    test_adata = adata[
        (adata.obs[celltype_column_name] == celltype_to_predict) & 
        (adata.obs[condition_column_name] == condition_to_predict)
    ].copy()

    # Select training set: all other cells
    train_adata = adata[
        ~((adata.obs[celltype_column_name] == celltype_to_predict) & 
          (adata.obs[condition_column_name] == condition_to_predict))
    ].copy()

    # Error checking
    if test_adata.shape[0] == 0:
        raise ValueError(f"No test cells found for cell type '{celltype_to_predict}' under condition '{condition_to_predict}'.")

    if train_adata.shape[0] == 0:
        raise ValueError("Training set is empty after filtering.")

    print(f'Train_adata shape: {train_adata.shape}')
    # Setup anndata
    train_adata = setup_anndata(train_adata, copy=True, batch_key=condition_column_name, labels_key=celltype_column_name)
    test_adata = setup_anndata(test_adata, copy=True, batch_key=condition_column_name, labels_key=celltype_column_name)
    print(f'Train_adata shape after setup_anndata: {train_adata.shape}')
    return train_adata, test_adata



def calculate_r2_singledose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys:dict,
    diff_genes = None,
    random_sample_coef = None,
    n_iter: int = 1,
):
    #Set condition key
    
    #Densify sparse matrix
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    
    #Treatment and Prediction
    treat = adata[adata.obs[condition_key] == axis_keys["y"]] 
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]
    
    r2_values_dict = {"R^2":[], "Gene Set":[]}
    for i in range(n_iter):
        
        if random_sample_coef is not None:
            treat_idx = np.random.choice(np.arange(0, treat.shape[0]), int(random_sample_coef*treat.shape[0]))
            pred_idx = np.random.choice(np.arange(0, pred.shape[0]), int(random_sample_coef*pred.shape[0]))
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]
        
        else:
            treat_samp = treat
            pred_samp = samp
            
        if diff_genes is not None:
            treat_diff = treat_samp[:, diff_genes]
            pred_diff = pred_samp[:, diff_genes]

            x_diff = np.average(pred_diff.X, axis=0)
            y_diff = np.average(treat_diff.X, axis = 0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            r2_values_dict["R^2"].append(r_value_diff ** 2)
            r2_values_dict["Gene Set"].append("DEGs")
            
    
        x = np.average(treat_samp.X, axis = 0)
        y = np.average(pred_samp.X, axis = 0)
    
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value ** 2)
        r2_values_dict["Gene Set"].append("All HVGs")
        
    r2_values_df = pd.DataFrame(r2_values_dict)
    r2_values_df["Cell"] = cell
    r2_values_df["Model"] = model
        
    return r2_values_df

def calculate_r2_multidose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys:dict,
    diff_genes = None,
    random_sample_coef = None,
    n_iter: int = 1,
):
    #Set condition key
    
    #Densify sparse matrix
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    
    #Treatment and Prediction
    treat = adata[adata.obs[condition_key] == axis_keys["y"]] 
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]
    
    r2_values_dict = {"R^2":[], "Gene Set":[]}
    for i in range(n_iter):
        
        if random_sample_coef is not None:
            treat_idx = np.random.choice(np.arange(0, treat.shape[0]), int(random_sample_coef*treat.shape[0]))
            pred_idx = np.random.choice(np.arange(0, pred.shape[0]), int(random_sample_coef*pred.shape[0]))
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]
        
        else:
            treat_samp = treat
            pred_samp = samp
            
        if diff_genes is not None:
            treat_diff = treat_samp[:, diff_genes]
            pred_diff = pred_samp[:, diff_genes]

            x_diff = np.average(pred_diff.X, axis=0)
            y_diff = np.average(treat_diff.X, axis = 0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            r2_values_dict["R^2"].append(r_value_diff ** 2)
            r2_values_dict["Gene Set"].append("DEGs")
            
    
        x = np.average(treat_samp.X, axis = 0)
        y = np.average(pred_samp.X, axis = 0)
    
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value ** 2)
        r2_values_dict["Gene Set"].append("All HVGs")
        
    r2_values_df = pd.DataFrame(r2_values_dict)
    r2_values_df["Cell"] = cell
    r2_values_df["Model"] = model
    r2_values_df["Dose"] = axis_keys["y"]
    return r2_values_df




# def proportional_sample(adata, key, target_total=5_000, min_cells=100):
#     pop_dict = Counter(adata.obs[key])
#     total = sum(pop_dict.values())
#     idxs = []

#     for group, count in pop_dict.items():
#         group_idx = np.where(adata.obs[key] == group)[0]
#         # Calculate sample size proportionally
#         n_sample = int((count / total) * target_total)
#         # Ensure at least `min_cells` if possible
#         n_sample = max(min_cells, n_sample) if len(group_idx) >= min_cells else len(group_idx)
#         # Sample without replacement (with safety fallback)
#         replace = len(group_idx) < n_sample
#         sampled_idx = np.random.choice(group_idx, n_sample, replace=replace)
#         idxs.append(sampled_idx)

#     # Flatten list of arrays and return subset
#     final_idxs = np.concatenate(idxs)
#     return adata[final_idxs].copy()





def random_sample(
		adata,
		key,
		max_or_min="max",
		replacement = True
		):
	"""
	Makes the populations of each cell group equal based on the property of interest. 
	Will take the populations of each group within the property of interest and take an equal random sample of each group.

	Parameters
	----------

	adata: anndata object to balance populations for
	key: property of interest to balance
	max_or_min: to make the maximum population size or minimum population size as the size we want all populations to be at
	replacement: sampling with or without replacement. If max_or_min == "max", we automatically sample with replacement.

	Return
	----------
	resampled_adata: AnnData
	an anndata object containing equal size populations for each group within the property of interest.
	"""
	#list of groups within property of interest and find their cell population sizes
	pop_dict = Counter(adata.obs[key])
	#Find whether we want to take the maximum or the minimum size of the population
	eq = np.max(list(pop_dict.values())) if max_or_min == "max" else np.min(list(pop_dict.values()))
	#replacement?
	replacement = True if max_or_min == "max" else replacement
	#Find the indexes for sampling
	idxs = []
	for group in pop_dict.keys():
		group_bool = np.array(adata.obs[key] == group)
		group_idx = np.nonzero(group_bool)[0]
		group_idx_resampled = group_idx[np.random.choice(len(group_idx), eq, replace = replacement)]
		idxs.append(group_idx_resampled)
	
	resampled_adata = adata[np.concatenate(idxs)].copy()
	return resampled_adata
