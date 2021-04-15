import os
import numpy as np
import pandas as pd

from src.RF import rf_config
from src.generalutils import helper

print('extract_vessel_voxel_data', os.getcwd())

########################################
# SET
########################################
feature_dtype = 'float32'
label_dtype = 'uint8'
########################################
df_column_names = rf_config.FEATURE_NAMES
class_dict = rf_config.CLASS_DICTS[rf_config.H_LEVEL]

# directories, paths and file names
feature_filenames = rf_config.FEATURE_FILENAMES
label_filename = rf_config.LABEL_FILENAMES[rf_config.H_LEVEL]
dataframe_savename = rf_config.DATAFRAME_FILE_NAME
data_dir = rf_config.DATA_DIR
patient_files = rf_config.PATIENTS['train']['working'] + rf_config.PATIENTS['val']['working'] + \
                rf_config.PATIENTS['test']['working']
print('Patient files:', patient_files)

# check number of patients in datasets
num_patients = len(patient_files)
print('Number of patients:', num_patients)
save_directory = os.path.join(data_dir, rf_config.VESSEL_VOXEL_DATA_DIR)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# extract features and labels voxel-wise from patient in train dataset
for patient in patient_files:
    print('PATIENT: ', patient)

    # load image, label and skeleton
    feature_list = [helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + f),
                                                    printout=True) for f in feature_filenames]
    label = helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + label_filename), printout=True)

    # check if the skeletons of the different features and target match
    helper.nonzero_indices_match(feature_list + [label])

    # find all vessel voxel indices
    vessel_inds = np.where(feature_list[0] > 0)
    num_voxels = len(vessel_inds[0])

    # prepare matrices for saving
    features = []
    labels = []

    # iterate over all vessel voxels and extract features and labels
    for voxel in range(len(vessel_inds[0])):
        x, y, z = vessel_inds[0][voxel], vessel_inds[1][voxel], vessel_inds[2][voxel]  # vessel voxel coordinates
        features.append([])
        features[voxel].append(patient)  # patient id
        features[voxel].append(x)  # x coordinate
        features[voxel].append(y)  # y coordinate
        features[voxel].append(z)  # z coordinate
        for i, feature in enumerate(feature_list):
            features[voxel].append(feature[x, y, z])
        labels.append(label[x, y, z])  # ground-truth label

    df = pd.DataFrame(data=features, columns=['patient_id'] + df_column_names)
    df['label'] = labels

    # save the dataframes for each patient
    save_path = os.path.join(save_directory, patient + dataframe_savename)
    df.to_pickle(save_path)
    print("Dataframe saved to ", save_path)
    print('---------------------------------------------------------------------')
print('DONE')
