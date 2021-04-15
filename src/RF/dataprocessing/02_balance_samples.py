"""
This scripts balances the given data sets to have the same number of samples for each class.
"""

import os
import pandas as pd
from src import config
from src.generalutils import helper

print('balance_samples', os.getcwd())

patients = config.PATIENTS
data_dir = config.DATA_DIR
dataframe_file_name = config.DATAFRAME_FILE_NAME
dataframe_save_name = config.BALANCED_DATAFRAME_FILE_NAME

vessel_voxel_data_dir = os.path.join(data_dir, config.VESSEL_VOXEL_DATA_DIR)
# get all patients (e.g. working + working augmented)
all_patients = config.PATIENTS['train']['working'] + config.PATIENTS['val']['working'] + \
               config.PATIENTS['test']['working']
num_patients = len(all_patients)
print('Number of patients', num_patients)

# read the voxel skeleton data for each patient and pool the data to one dataframe
dataframe_list = [pd.read_pickle(
    os.path.join(vessel_voxel_data_dir, patient + dataframe_file_name)) for patient in all_patients]
pooled_df = pd.concat(dataframe_list)
print('FULL DF')
print(pooled_df.describe(include='all'))
classes = pooled_df['label']

# find minimal voxel count
unique, count = helper.unique_and_count(classes, name='classes')
min_count = min(count)
print('minimal voxel count:', min_count)

# balance the classes to equal the minimal voxel count
grouped_df = pooled_df.groupby('label')
balanced_df = grouped_df.apply(lambda x: x.sample(n=min_count, replace=False))
print('BALANCED DF')
print(balanced_df.describe(include='all'))
print(balanced_df.dtypes)

# save the balanced dataframe into one file for each dataset
save_path = os.path.join(vessel_voxel_data_dir, dataframe_save_name)
balanced_df.to_pickle(save_path)
print("Dataframe saved to ", save_path)

print('DONE')
