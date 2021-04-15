import os

import pandas as pd

from src.RF import rf_config


def get_balanced_samples(feature_names=None):
    rf_data_dirs = os.path.join(rf_config.DATA_DIR, rf_config.VESSEL_VOXEL_DATA_DIR)
    df = pd.read_pickle(os.path.join(rf_data_dirs, rf_config.BALANCED_DATAFRAME_FILE_NAME))
    df = df[['patient_id'] + feature_names + ['label']]
    print("df head \n", df.head())
    print("df shape", df.shape)

    y = df[['patient_id', 'label']]
    X = df.drop('label', axis=1)
    print("y head \n", y.head())
    print("y shape", y.shape)
    print("X head \n", X.head())
    print("X shape", X.shape)

    return X, y
