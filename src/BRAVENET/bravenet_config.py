"""
This file contains constants like paths, names, sizes and general parameters related only to the
BRAVENET sub-project. It imports the general config from src.config.py.
"""
from datetime import datetime

from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from src.config import *
from src.generalutils.metrics import dice_multiclass

print('bravenet_config', os.getcwd())
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# -----------------------------------------------------------
# VERSION SETTINGS
# -----------------------------------------------------------
IMAGE_SPECS = 'raw'
PATCH_SIZE_X = 128
PATCH_SIZE_Y = 128
PATCH_SIZE_Z = 64
PATCHES_DIR0 = '3dpatches_' + IMAGE_SPECS
PATCHES_DIR1 = '8p' + str(PATCH_SIZE_X) + 'x' + str(PATCH_SIZE_Y) + 'x' + str(
    PATCH_SIZE_Z) + '_rad_' + H_LEVEL
PATCHES_DIR = os.path.join(PATCHES_DIR0, PATCHES_DIR1)
SAMPLES_PATH = os.path.join(DATA_DIR, PATCHES_DIR)
NET_VERSION = 'bravenet_' + PATCHES_DIR1 + ''  # Net version name.
FEATURE_NAMES = ['tof_mra', 'radius_skeleton']
FEATURE_FILENAMES = ['_img.nii.gz', '_skel_radius.nii.gz']
MAX_FEATURE_VALUES = [1355.0, 3.7004213]  # for original train dataset, not for crossvalidation folds

# -----------------------------------------------------------
# GENERAL MODEL PARAMETERS
# -----------------------------------------------------------
NUM_FEATURES = len(FEATURE_NAMES)  # number of channels in input data
ACTIVATION = 'relu'  # activation_function after every convolution
FINAL_ACTIVATION = 'softmax'  # activation_function of the final layer
LOSS_FUNCTION = sparse_categorical_crossentropy
METRICS = ['acc',
           dice_multiclass(smooth=0, ignore_background=0, n_classes=NUM_CLASSES)]  # class averaged dice coefficient
OPTIMIZER = Adam  # Adam: algorithm for first-order gradient-based optimization of stochastic objective functions

# -----------------------------------------------------------
# DIRECTORIES, PATHS AND FILE NAMES
# -----------------------------------------------------------
# directories for saving models, predictions and results
ARCHITECTURE = NET_VERSION
RESULTS_DIR = os.path.join(TOP_LEVEL_DATA, 'results/' + IMAGE_SPECS)
MODELS_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'models')
LOG_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'logs', 'scalars', datetime.now().strftime("%Y%m%d-%H%M%S"))
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'predictions')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, ARCHITECTURE, 'visualization')

# title for saving the results of one training run
def get_run_name(version, n_epochs, batch_size, learning_rate, dropout_rate, l1, l2, batch_normalization, deconvolution,
                 n_base_filters, n_patients_train, n_patients_val, formatting_epoch=False):
    run_name = ''
    if len(version) > 0:
        run_name = '_' + version
    if formatting_epoch:
        run_name += '_ep' + '{epoch}'
    else:
        run_name += '_ep' + str(n_epochs)
    run_name += 'bs' + str(batch_size) + 'lr' + str(learning_rate)
    if dropout_rate:
        run_name += 'dr' + str(dropout_rate)
    if l1:
        run_name += 'L1_' + str(l1)
    if l2:
        run_name += 'L2_' + str(l2)
    if batch_normalization:
        run_name += 'bn'
    if deconvolution:
        run_name += '_deconv'
    run_name += '_filt' + str(n_base_filters) + 'train' + str(n_patients_train) + 'val' + str(n_patients_val)
    print('RUN', run_name)
    return run_name
