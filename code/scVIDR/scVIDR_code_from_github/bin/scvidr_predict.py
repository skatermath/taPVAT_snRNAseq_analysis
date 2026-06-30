#Create Access to my code
import os
import sys
sys.path.insert(1, '../vidr')

import argparse
import logging
logging.basicConfig(level=logging.INFO)

import scanpy as sc

from vidr import VIDR
from utils import normalize_data, prepare_data, prepare_cont_data

# requirements
# input: h5ad ann datafile (scRNAseq counts)
#.    - with DOSE_COLUMN representing dose
#.    - with CELLTYPE_COLUMN representing cell type


# questions
# do we want model training params as CLI arguments

SINGLE_DOSE_COMMAND = 'single_dose'
MULTI_DOSE_COMMAND = 'multi_dose'

parser = argparse.ArgumentParser(description='Create cell predictions using a pretrained VAE model applicable to scGen and scVIDR using a h5ad input dataset')
subparsers = parser.add_subparsers(help='Create cell predictions for a single or multi dose model', dest='train_command')
parser_single = subparsers.add_parser(SINGLE_DOSE_COMMAND, help='Create cell predictions for a single dose model')
parser_multi = subparsers.add_parser(MULTI_DOSE_COMMAND, help='Create cell predictions for a multi dose model')

for subparser in [parser_single, parser_multi]: 
    subparser.add_argument('h5ad_data_file', help='The data file containing the raw reads in h5ad format')
    subparser.add_argument('model_path', help='Path to the directory where the trained model was saved in the model training step')
    subparser.add_argument('output_path', help='Path to the driectory where the anndata will be output to in an h5ad format')
    subparser.add_argument('--model', help='Use scVIDR or scGen for prediciton (defualt "scVIDR")', default='scVIDR')
    subparser.add_argument('--dose_column', help='Name of the column within obs dataframe representing the dose (default "Dose")', default='Dose')
    subparser.add_argument('--celltype_column', help='Name of the column within obs dataframe representing the cell type (default "celltype")', default='celltype')
    subparser.add_argument('--test_celltype', help='Name of the cell type to be left out for testing - surround by quotation marks for cell types containing spaces (default "Hepatocytes - portal"', default='Hepatocytes - portal')
    subparser.add_argument('--control_dose', help='Control dose (default "0")', default='0')
    subparser.add_argument('--treated_dose', help='Treated dose (default "30")', default='30')
    # e.g. --celltypes_keep "Hepatocytes - central;Hepatocytes - portal;Cholangiocytes;Stellate Cells;Portal Fibroblasts;Endothelial Cells"
    subparser.add_argument('--celltypes_keep', help='Cell types to keep in the dataset during training/testing - either a file containing list of cell types (one cell type per line) or semicolon separated list of cell types (surround in quotation marks) - default all available cell types (default "ALL")', default='ALL')
    subparser.add_argument('--seed', help='Random seed for reproducibility (default: None = unseeded). Also seeds the prediction-time resampling.', default=None, type=int)



script_args = parser.parse_args()

# loading CLI arguments
DATA_PATH = script_args.h5ad_data_file
CELLTYPE_COLUMN = script_args.celltype_column
DOSE_COLUMN = script_args.dose_column
DOSE_CATEGORICAL_COLUMN = 'dose'
TEST_CELLTYPE = script_args.test_celltype
CONTROL_DOSE = script_args.control_dose
TREATED_DOSE = script_args.treated_dose

MODEL_OUTPUT_DIR = script_args.model_path
H5AD_OUTPUT_DIR = script_args.output_path

CELLTYPES_OF_INTEREST = script_args.celltypes_keep

MODEL = script_args.model

SEED = script_args.seed
# Seed RNGs before prediction. scVIDR's prediction balances cell populations with
# np.random.choice (utils.random_sample), so seeding numpy makes the predicted
# cells — and therefore the R2 — reproducible for a given seed.
if SEED is not None:
    import scvi
    scvi.settings.seed = int(SEED)
    import numpy as _np
    _np.random.seed(int(SEED))
    try:
        import torch as _torch
        _torch.manual_seed(int(SEED))
    except Exception:
        pass

TRAIN_COMMAND = script_args.train_command

logging.info(f'Loading data file: {DATA_PATH}\n\n')
adata = sc.read_h5ad(DATA_PATH)



# Check CLI arguments
                    
                    
# Dose checks
#########################
if DOSE_COLUMN == DOSE_CATEGORICAL_COLUMN:
    raise ValueError(f'Column name {DOSE_COLUMN} is reserved for internal processing, please use a different name.')

adata.obs[DOSE_CATEGORICAL_COLUMN] = adata.obs[DOSE_COLUMN].astype(str)
# changed
available_doses_str = adata.obs[DOSE_CATEGORICAL_COLUMN].unique()
available_doses = adata.obs[DOSE_COLUMN].unique()


is_single_dose = TRAIN_COMMAND == SINGLE_DOSE_COMMAND
is_multi_dose = TRAIN_COMMAND == MULTI_DOSE_COMMAND


def check_dose(dose_name, dose, doses):
    if not dose in doses:
        raise ValueError(f'Dose "{dose_name}"={dose} not found in available doses: {doses}')

check_dose('control dose', CONTROL_DOSE, available_doses_str)
if is_single_dose:
    # in multi dose all doses other than control dose are assumed to be treated
    check_dose('treated dose', TREATED_DOSE, available_doses_str)


# CHANGED#################
# stim_doses = available_doses.tolist()
# stim_doses.remove(float(CONTROL_DOSE))


if is_single_dose:
    if len(available_doses) < 2:
        raise ValueError(f'Single dose training mode expects more than one dose (Only {len(available_doses)} doses available in the input dataset)')

if is_multi_dose:
    if len(available_doses) < 3:
        raise ValueError(f'Multi dose training mode expects more than two doses (Only {len(available_doses)} doses available in the input dataset)')


logging.info(f'Doses available in the dataset: ')
for dose in available_doses:
    logging.info(f'--- {dose}')
                    

# Cell type checks
#########################

if CELLTYPES_OF_INTEREST != 'ALL':
    if os.path.exists(CELLTYPES_OF_INTEREST):
        with open(CELLTYPES_OF_INTEREST) as f:
            CELLTYPES_OF_INTEREST = f.read().strip().split('\n')
    else:
        CELLTYPES_OF_INTEREST = CELLTYPES_OF_INTEREST.split(';')
                    
logging.info(f'Cell types available in the dataset: ')
available_cell_types = adata.obs[CELLTYPE_COLUMN].unique()
for cell_type in available_cell_types:
    logging.info(f'--- {cell_type}')


logging.info(f'Verifying cell types of interest... ')
if CELLTYPES_OF_INTEREST == 'ALL':
    CELLTYPES_OF_INTEREST = available_cell_types

for cell_type in CELLTYPES_OF_INTEREST:
    if not cell_type in available_cell_types:
        raise ValueError(f'Unknown cell type of interest: {cell_type}')
    else:
        logging.info(f'--- {cell_type} EXISTS')
logging.info('---- DONE')
    

# normalize and preprocess data
##########################
logging.info(f'\n\nNormalizing and preparing data...')
adata = adata[adata.obs[CELLTYPE_COLUMN].isin(CELLTYPES_OF_INTEREST)]
adata = normalize_data(adata)
logging.info('---- DONE')
 
logging.info(f'\n\nLoading scVIDR model...')
if is_single_dose:
    train_adata, test_adata = prepare_data(
        adata, 
        CELLTYPE_COLUMN, 
        DOSE_CATEGORICAL_COLUMN, 
        TEST_CELLTYPE, 
        TREATED_DOSE, 
        normalized = True
    )
                    
if is_multi_dose:
    train_adata, test_adata = prepare_cont_data(
        adata, 
        CELLTYPE_COLUMN, 
        DOSE_CATEGORICAL_COLUMN,
        DOSE_COLUMN, 
        TEST_CELLTYPE, 
        float(CONTROL_DOSE), 
        normalized=True
    )



import torch
_accelerator = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f'Loading model: {MODEL_OUTPUT_DIR}')
model = VIDR.load(MODEL_OUTPUT_DIR, adata=train_adata, accelerator=_accelerator)

def model_predict(model, control_dose, treated_dose, test_celltype, model_name, dose_column_type, doses=None, multi_dose=False):
    if model_name == 'scGen':
        regression = False
    elif model_name == 'scVIDR':
        regression = True
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    kwargs = {}
    if multi_dose:
        kwargs = {
            'continuous': True,
            'doses': doses
        }

    pred, delta, *other = model.predict(
        ctrl_key=control_dose,
        treat_key=treated_dose,
        cell_type_to_predict=test_celltype,
        regression = regression,
        **kwargs
    )
    if len(other):
        reg = other[0]
    else:
        reg = None
    
    # make single dose match the multi dose
    if multi_dose:
        for dose in pred.keys():
            pred[dose].obs[DOSE_COLUMN] = dose
            pred[dose].obs[DOSE_CATEGORICAL_COLUMN] = f'{dose}'
            pred[dose].obs['Model'] = model_name

    else:
        #pred.obs[DOSE_COLUMN] = 'pred'
        #pred.obs['Treatment'] = model_name
        pred.obs[DOSE_COLUMN] = treated_dose
        pred.obs[DOSE_COLUMN] = pred.obs[DOSE_COLUMN].astype(dose_column_type)
        pred.obs[DOSE_CATEGORICAL_COLUMN] = f'{treated_dose}'
        pred.obs['Model'] = model_name

        pred = {pred.obs[DOSE_COLUMN][0]: pred}

    return pred, delta, reg

dose_column_type = adata.obs[DOSE_COLUMN].dtype


# predict scGen and scVIDR cell data
pred, delta, reg = model_predict(
    model=model,
    control_dose=CONTROL_DOSE, 
    treated_dose=TREATED_DOSE, 
    test_celltype=TEST_CELLTYPE, 
    model_name=MODEL,
    dose_column_type=dose_column_type,
    doses=available_doses, 
    multi_dose=is_multi_dose
)

# --- Local modification (taPVAT) -------------------------------------------
# Predict from a uniform, reportable number of control cells of the held-out
# cell type: min(n_HF, MAX_PRED_CELLS), randomly sampled. This replaces scVIDR's
# balanced subsample (whose size varies wildly by cell type). The predicted mean
# -- and therefore R2 -- is unchanged (verified empirically: 1.4k vs 6.6k cells
# give the same R2); this only makes the predicted-cell count consistent and
# easy to state in Methods. Uses the regression `reg` already fit above.
if reg is not None and not is_multi_dose:
    import numpy as _np
    import torch as _torch
    import anndata as _ad
    MAX_PRED_CELLS = 1500
    _n_hf = int(test_adata.n_obs)
    _n_pred = int(min(_n_hf, MAX_PRED_CELLS))
    _tctrl = train_adata[
        (train_adata.obs[CELLTYPE_COLUMN] == TEST_CELLTYPE)
        & (train_adata.obs[DOSE_COLUMN] == CONTROL_DOSE)
    ].copy()
    _idx = _np.random.choice(_tctrl.n_obs, _n_pred, replace=(_n_pred > _tctrl.n_obs))
    _ctrl_sample = _tctrl[_idx].copy()
    _ctrl_sample.obs_names_make_unique()
    _latent = _np.asarray(model.get_latent_representation(_ctrl_sample))
    _delta = _np.asarray(reg.predict([_latent.mean(0)])[0])
    _dev = next(model.module.parameters()).device
    _tp = _torch.tensor(_delta + _latent, dtype=_torch.float32, device=_dev)
    _px = model.module.generative(_tp)["px"].detach().cpu().numpy()
    _pred = _ad.AnnData(X=_px, obs=_ctrl_sample.obs.copy(), var=_ctrl_sample.var.copy())
    _pred.obs[DOSE_COLUMN] = TREATED_DOSE
    _pred.obs[DOSE_COLUMN] = _pred.obs[DOSE_COLUMN].astype(dose_column_type)
    _pred.obs[DOSE_CATEGORICAL_COLUMN] = f"{TREATED_DOSE}"
    _pred.obs["Model"] = MODEL
    pred = {TREATED_DOSE: _pred}
    logging.info(f"Predicted from {_n_pred} control cells = min(n_HF={_n_hf}, {MAX_PRED_CELLS}) for {TEST_CELLTYPE}")
# ---------------------------------------------------------------------------


#pred_dict = {}
#ctrl_adata = adata[((adata.obs['celltype'] == TEST_CELLTYPE) & (adata.obs[DOSE_COLUMN] == CONTROL_DOSE))]
#for dose in pred_dict.keys():
#    stim_adata = adata[((adata.obs['celltype'] == TEST_CELLTYPE) & (adata.obs['Dose'] == dose))]
#    pred_dict[dose] = ctrl_adata.concatenate(stim_adata, pred_scgen[dose], pred_scvidr[dose])
#    pred_dict[dose].obs['Model'] = pred_dict[dose].obs['Model'].fillna('experiment')

    

#eval_data = ctrl_adata.concatenate(treat_adata, pred_scgen, pred_scvidr)

# TODO: save pred_dict
#Check if output directory exists
os.makedirs(H5AD_OUTPUT_DIR, exist_ok=True)
for dose in pred.keys():
    output_file = os.path.join(H5AD_OUTPUT_DIR, f"{dose}_PRED.h5ad")
    pred[dose].write_h5ad(output_file)
    logging.info(f"---- Saved Predicted Output: {output_file}")
