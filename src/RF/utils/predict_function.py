"""
This script contains the function that extracts features per voxel for prediction with RF and than reconstructs the 3D
volume again.
"""

import os
import pickle
import sys

import numpy as np

from src.RF import rf_config
from src.generalutils import helper


def predict_and_save(version, n_estimators,
                     n_patients_train, patient, data_dir, dataset, feature_filenames, models_dir, predictions_dir,
                     run_name=None, model=None, train_metadata=None):
    """
    :param version: The name describes the SVM version. String.
    :param n_estimators: Number of trees in forest. Positive integer.
    :param n_patients_train: The number of patient used for training. Positive integer.
    :param patient: The patient name. String.
    :param data_dir: The path to data dirs. String.
    :param dataset: The dataset. E.g. train/val/set
    :param feature_filenames: List of file names of the feature inputs to the network. List of strings.
    :param models_dir: Directory where trained models are saved. String.
    :param predictions_dir: Directory where predictions are saved. String.
    """
    print('model version', version)
    print('number of estimators', n_estimators)
    print('patient:', patient)
    print('feature file names', feature_filenames)

    # create the name of current run
    if not run_name:
        run_name = rf_config.get_run_name(version=version, n_estimators=n_estimators, n_patients_train=n_patients_train)

    # -----------------------------------------------------------
    # LOADING MODEL, RESULTS AND WHOLE BRAIN MATRICES
    # -----------------------------------------------------------
    try:
        if not model:
            model_filepath = rf_config.get_model_filepath(models_dir, run_name)
            print('Model path:', model_filepath)
            model = pickle.load(open(model_filepath, 'rb'))
        # -----------------------------------------------------------
        # TRAINING PARAMS AND FEATURES
        # -----------------------------------------------------------
        if not train_metadata:
            try:
                train_metadata_filepath = rf_config.get_train_metadata_filepath(models_dir, run_name)
                with open(train_metadata_filepath, 'rb') as handle:
                    train_metadata = pickle.load(handle)
                print('Train params:')
                print(train_metadata['params'])
            except FileNotFoundError:
                print('Unexpected error by reading train metadata:', sys.exc_info()[0])

        print('> Loading features...')
        loaded_feature_list = [helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + f),
                                                               printout=True) for f in feature_filenames]

        # -----------------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------------
        print('Predicting...')
        starttime_predict = helper.start_time_measuring(what_to_measure='patient prediction')

        # find all vessel voxel indices
        vessel_inds = np.where(loaded_feature_list[0] > 0)

        # extract features and labels per voxel and predict
        prediction_3D = np.zeros(loaded_feature_list[0].shape, dtype='uint8')
        for voxel in range(len(vessel_inds[0])):
            x, y, z = vessel_inds[0][voxel], vessel_inds[1][voxel], vessel_inds[2][voxel]  # vessel voxel coordinates
            features = [[x, y, z]]
            for i, feature in enumerate(loaded_feature_list):
                features[0].append(feature[x, y, z])

            # scale data
            scaler = train_metadata['params']['scaler']
            features = scaler.transform(features)

            # predict
            prediction = model.predict(features)

            # rebuild the 3D volume
            prediction_3D[x, y, z] = prediction

        # how long does the prediction take for a patient
        helper.end_time_measuring(starttime_predict, what_to_measure='patient prediction')

        # -----------------------------------------------------------
        # SAVE AS NIFTI
        # -----------------------------------------------------------
        print(predictions_dir)
        save_path = rf_config.get_annotation_filepath(predictions_dir, run_name, patient, dataset)
        helper.create_and_save_nifti(prediction_3D, save_path)
    except FileNotFoundError:
        print('Unexpected error by reading model:', sys.exc_info()[0])
