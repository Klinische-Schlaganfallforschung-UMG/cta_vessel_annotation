"""
This file contains general constants like paths, names, sizes and general parameters for all sub-project used in
this project.
"""

import os
import time

import pandas as pd
from matplotlib import colors
from matplotlib.colors import ListedColormap

from src.generalutils import helper

print('config', os.getcwd())
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

# comment and uncomment following 2 lines according to where you run the program
# TOP_LEVEL_DATA = '.'
TOP_LEVEL_DATA = '../work/'

# -----------------------------------------------------------
# VERSION SETTINGS
# -----------------------------------------------------------
H_LEVEL = 'coarse0'  # hierchical level: 'coarse0', 'fine'
# dictionary for labels for hierarchical levels
LABEL_FILENAMES = {
    'fine': '_skel_ann_fine.nii.gz',
    'coarse0': '_skel_ann_coarse0.nii.gz'
}
# dictionary of all possible features
ALL_FEATURE_FILENAMES = {
    'tof_mra': '_img.nii.gz',
    'radius_skeleton': '_skel_radius.nii.gz',
    'segmented_skeleton_washed_in_y': '_skel_seg_y.nii.gz'
}

# -----------------------------------------------------------
# PATIENT DICTIONARY
# -----------------------------------------------------------
# matching the patient to train/val/test set, conversion from original patient names to working names
# according to conversion table: ./docs/file_naming.xlsx
PATIENTS = helper.patient_dict(file='docs/file_naming.xlsx', sheetname='data_202003', database=None)

# -----------------------------------------------------------
# CLASS DICTIONARY
# -----------------------------------------------------------
CLASS_DICTS = {
    'fine': {0: ['background'],
             1: ['not-annotated'],
             2: [[2, 'DEXTER'], [30, 'DEXTER'], [31, 'DEXTER']],  # ICA
             3: [[2, 'SINISTER'], [30, 'SINISTER'], [31, 'SINISTER']],  # ICA
             4: [[3, 'DEXTER'], [33, 'DEXTER'], [34, 'DEXTER']],  # ICA CoW
             5: [[3, 'SINISTER'], [33, 'SINISTER'], [34, 'SINISTER']],  # ICA CoW
             6: [[4, 'DEXTER'], [36, 'DEXTER'], [37, 'DEXTER']],  # M1
             7: [[4, 'SINISTER'], [36, 'SINISTER'], [37, 'SINISTER']],  # M1
             8: [[5, 'DEXTER'], [39, 'DEXTER']],  # M2 sup
             9: [[5, 'SINISTER'], [39, 'SINISTER']],  # M2 sup
             10: [[6, 'DEXTER'], [42, 'DEXTER']],  # M2 inf
             11: [[6, 'SINISTER'], [42, 'SINISTER']],  # M2 inf
             12: [[11, 'DEXTER'], [57, 'DEXTER'], [58, 'DEXTER']],  # A1
             13: [[11, 'SINISTER'], [57, 'SINISTER'], [58, 'SINISTER']],  # A1
             14: [[12, 'MEDIAN']],  # AcomA
             15: [[13, 'DEXTER']],  # A2
             16: [[13, 'SINISTER']],  # A2
             17: [[16, 'DEXTER'], [72, 'DEXTER'], [73, 'DEXTER']],  # VA
             18: [[16, 'SINISTER'], [72, 'SINISTER'], [73, 'SINISTER']],  # VA
             19: [[17, 'MEDIAN'], [76, 'MEDIAN']],  # BA
             20: [[18, 'DEXTER']],  # PcomA
             21: [[18, 'SINISTER']],  # PcomA
             22: [[19, 'DEXTER'], [81, 'DEXTER'], [82, 'DEXTER']],  # P1
             23: [[19, 'SINISTER'], [81, 'SINISTER'], [81, 'SINISTER']],  # P1
             24: [[20, 'DEXTER'], [85, 'DEXTER']],  # P2
             25: [[20, 'SINISTER'], [85, 'SINISTER']]  # P2
             },
    'coarse0': {0: ['background'],
                1: ['not-annotated'],
                # ICA, ICA CoW
                2: [[2, 'DEXTER'], [30, 'DEXTER'], [31, 'DEXTER'], [3, 'DEXTER'], [33, 'DEXTER'], [34, 'DEXTER']],
                3: [[2, 'SINISTER'], [30, 'SINISTER'], [31, 'SINISTER'], [3, 'SINISTER'], [33, 'SINISTER'],
                    [34, 'SINISTER']],
                # M1, M2 sup, M2 inf
                4: [[4, 'DEXTER'], [36, 'DEXTER'], [37, 'DEXTER'], [5, 'DEXTER'], [39, 'DEXTER'], [6, 'DEXTER'],
                    [42, 'DEXTER']],
                5: [[4, 'SINISTER'], [36, 'SINISTER'], [37, 'SINISTER'], [5, 'SINISTER'], [39, 'SINISTER'],
                    [6, 'SINISTER'], [42, 'SINISTER']],
                # A1, A2
                6: [[11, 'DEXTER'], [57, 'DEXTER'], [58, 'DEXTER'], [13, 'DEXTER']],
                # A1, A2, AcomA
                7: [[11, 'SINISTER'], [57, 'SINISTER'], [58, 'SINISTER'], [12, 'MEDIAN'], [13, 'SINISTER']],
                # VA
                8: [[16, 'DEXTER'], [72, 'DEXTER'], [73, 'DEXTER']],
                9: [[16, 'SINISTER'], [72, 'SINISTER'], [73, 'SINISTER']],
                # BA
                10: [[17, 'MEDIAN'], [76, 'MEDIAN']],
                # PcomA, P1, P2
                11: [[18, 'DEXTER'], [19, 'DEXTER'], [81, 'DEXTER'], [82, 'DEXTER'], [20, 'DEXTER'], [85, 'DEXTER']],
                12: [[18, 'SINISTER'], [19, 'SINISTER'], [81, 'SINISTER'], [82, 'SINISTER'], [20, 'SINISTER'],
                     [85, 'SINISTER']],
                }
}
ARTER_DICTS = {
    'fine': {helper.stringify(artery): cls for cls, arteries in CLASS_DICTS['fine'].items() for artery in arteries},
    'coarse0': {helper.stringify(artery): cls for cls, arteries in CLASS_DICTS['coarse0'].items() for artery in
                arteries}
}

CLASS_FINE_TO_CLASS_COARSE0_DICT = {
    fine_class: ARTER_DICTS['coarse0'][artery] for artery, fine_class in ARTER_DICTS['fine'].items()
}

# -----------------------------------------------------------
# GENERAL MODEL PARAMETERS
# -----------------------------------------------------------
NUM_CLASSES = len(CLASS_DICTS[H_LEVEL])  # number of class labels
VOLUME_DIMENSIONS = (312, 384, 127)
VOXEL_SIZE = (0.5208333, 0.5208333, 0.65)

# -----------------------------------------------------------
# DIRECTORIES, PATHS AND FILE NAMES
# -----------------------------------------------------------
# directory where the original scans are stored
ORIGINAL_DATA_DIR = os.path.join(TOP_LEVEL_DATA, 'data', 'original_data')
# original files with scans
IMG_FILENAME = '00?.nii.gz'
SKEL_RADIUS_FILENAME = '001_Vessel-Manual-Gold-int_SKEL-RADIUS.nii.gz'
GRAPH_FILENAME = '001_Vessel-Manual-Gold-int_SKEL-RADIUS*.graph.json'
ANNOTATION_FILENAME = '001_Vessel-Manual-Gold-int_SKEL-RADIUS*.annotation.json'
# directories where masked and augmented data are stored
DATA_DIR = os.path.join(TOP_LEVEL_DATA, 'annotation_data')
# vessel voxel data
VESSEL_VOXEL_DATA_DIR = 'vessel_voxel_data_' + H_LEVEL  # folder for vessel voxel data
DATAFRAME_FILE_NAME = '_df.pkl'  # save name for saving svm data per patient in panda dataframe
BALANCED_DATAFRAME_FILE_NAME = 'balanced_df.pkl'


# where to store the trained model
def get_model_filepath(models_dir, run_name):
    return os.path.join(models_dir, 'model' + run_name + '.h5')


# where to store the results of training with parameters and training history
def get_train_metadata_filepath(models_dir, run_name, fold=None):
    if fold:
        return os.path.join(models_dir, fold, 'metadata' + run_name + '.pkl')
    return os.path.join(models_dir, 'metadata' + run_name + '.pkl')


# where to store csv file with training history
def get_train_history_filepath(models_dir, run_name):
    return os.path.join(models_dir, 'history' + run_name + '.csv')


# where to save predicted annotation as nifti
def get_annotation_filepath(predictions_dir, run_name, patient, dataset):
    return os.path.join(predictions_dir, dataset, patient + run_name + '.nii.gz')


# where to save predicted annotation washed with segments as nifti
def get_washed_annotation_filepath(predictions_dir, run_name, patient, dataset, washing_version=''):
    if washing_version == '':
        return os.path.join(predictions_dir, dataset, patient + run_name + '_wash.nii.gz')
    return os.path.join(predictions_dir, dataset, patient + run_name + '_wash_' + washing_version + '.nii.gz')


def get_tuned_parameters(models_dir):
    return os.path.join(models_dir, 'tuned_params.csv')


# where to save complete result table as csv
def get_complete_result_table_filepath(predictions_dir, dataset, washed=False, segment_wise=False):
    if washed:
        if segment_wise:
            return os.path.join(predictions_dir,
                                'result_table_' + dataset + '_washed_segment_wise_' + time.strftime("%Y%m%d-%H%M%S") +
                                '.csv')
        return os.path.join(predictions_dir, 'result_table_' + dataset + '_washed_' + time.strftime("%Y%m%d-%H%M%S") +
                            '.csv')
    return os.path.join(predictions_dir, 'result_table_' + dataset + '_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')


# where to save complete result table per patient as csv
def get_complete_result_table_per_patient_filepath(predictions_dir, dataset, washed=False, segment_wise=False):
    if washed:
        if segment_wise:
            return os.path.join(predictions_dir,
                                'result_table_per_patient_' + dataset + '_washed_segment_wise_' +
                                time.strftime("%Y%m%d-%H%M%S") + '.csv')
        return os.path.join(predictions_dir,
                            'result_table_per_patient_' + dataset + '_washed_' + time.strftime("%Y%m%d-%H%M%S") +
                            '.csv')
    return os.path.join(predictions_dir,
                        'result_table_per_patient_' + dataset + '_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')


# -----------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------
CUSTOM_CMAPS = {
    'fine': ListedColormap(
        ['black', 'white', 'orange', 'lightskyblue', 'sandybrown', 'cyan', 'magenta', 'limegreen', 'plum', 'blue',
         'red', 'green', 'pink', 'deepskyblue', 'yellow', 'firebrick', 'lightseagreen', 'hotpink', 'greenyellow',
         'gold', 'peachpuff', 'lime', 'lightsalmon', 'cornflowerblue', 'violet', 'palegreen']),
    'coarse0': ListedColormap(
        ['black', 'white', 'orange', 'lightskyblue', 'sandybrown', 'cyan', 'magenta', 'limegreen', 'plum', 'blue',
         'yellow', 'red', 'green'])
}

ERRORMAP_COLORS = {
    'TN': [0, 'black'],
    'TP': [1, 'red'],
    'FP': [2, 'green'],
    'FN': [3, 'blue'],
    'no_interest': [4, 'white']
}


def cmap_norm():
    cmap = CUSTOM_CMAPS[H_LEVEL]
    bounds = range(NUM_CLASSES + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm
