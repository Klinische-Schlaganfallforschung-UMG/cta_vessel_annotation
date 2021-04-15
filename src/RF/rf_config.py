"""
This file contains constants like paths, names, sizes and general parameters related only to the RF sub-project. It
imports the general config from src.config.py.
"""

from src.config import *

print('rf_config', os.getcwd())
pd.set_option('display.width', 500)

# -----------------------------------------------------------
# VERSION SETTINGS
# -----------------------------------------------------------
RF_VERSION = 'balanced'  # The rf version name. E.g.
FEATURE_NAMES = ['x', 'y', 'z', 'radius_skeleton']
FEATURE_FILENAMES = ['_skel_radius.nii.gz']

# -----------------------------------------------------------
# GENERAL MODEL PARAMETERS
# -----------------------------------------------------------
NUM_FEATURES = len(FEATURE_NAMES)  # number of channels in input data

# -----------------------------------------------------------
# DIRECTORIES, PATHS AND FILE NAMES
# -----------------------------------------------------------
# directories for saving models, predictions and results
ARCHITECTURE = 'RF' + RF_VERSION + '_' + H_LEVEL
RESULTS_DIR = os.path.join(TOP_LEVEL_DATA, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'models', str(NUM_FEATURES) + 'feature')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'predictions', str(NUM_FEATURES) + 'feature')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'visualization')


# title for saving the results of one training run
def get_run_name(version, n_estimators, n_patients_train):
    run_name = ''
    if len(version) > 0:
        run_name = '_' + version
    run_name += '_rf_n_estimators_' + str(n_estimators) + '_train_' + str(n_patients_train)
    print('RUN', run_name)
    return run_name


# where to store the trained model
def get_model_filepath(models_dir, run_name):
    return os.path.join(models_dir, 'model' + run_name + '.pkl')
