"""
This script contains a function that trains a model with given parameters and saves it.
"""
import os
import pickle

from sklearn.ensemble import RandomForestClassifier

from src.RF import rf_config
from src.generalutils import helper


def train_and_save(version, models_dir, n_estimators=10,
                   n_train_patients=0, train_X=None, train_y=None, scaler=None, run_name=None,
                   train_metadata_filepath=None, model_filepath=None):
    """
    :param version: The name describes the Unet version. String.
    :param models_dir: Directory where trained models are saved. String.
    :param n_estimators: Number of trees in forest. Positive integer.
    :param n_train_patients: The number of training patients. Positive integer.
    :param train_X: The features in train set. Ndarray.
    :param train_y: The labels in train set. Ndarray.
    :param scaler: Scikit scaler used for train set scaling.
    """
    print('________________________________________________________________________________')
    print('network version', version)
    print('number of estimators', n_estimators)
    print('num train samples', len(train_X))

    # -----------------------------------------------------------
    # CREATING NAME OF CURRENT RUN
    # -----------------------------------------------------------
    if not run_name:
        run_name = rf_config.get_run_name(version=version, n_estimators=n_estimators,
                                      n_patients_train=n_train_patients)
    # file paths
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not train_metadata_filepath:
        train_metadata_filepath = rf_config.get_train_metadata_filepath(models_dir, run_name)
    if not model_filepath:
        model_filepath = rf_config.get_model_filepath(models_dir, run_name)

    # -----------------------------------------------------------
    # CREATING MODEL
    # -----------------------------------------------------------
    print('Creating new model in', model_filepath)
    print('Training Random forest...')
    model = RandomForestClassifier(n_estimators=n_estimators)

    # -----------------------------------------------------------
    # TRAINING MODEL
    # -----------------------------------------------------------
    starttime_train = helper.start_time_measuring(what_to_measure='training')
    model.fit(train_X, train_y)
    print(model)
    endtime_train, duration_train = helper.end_time_measuring(starttime_train, what_to_measure='training')

    # -----------------------------------------------------------
    # SAVING MODEL
    # -----------------------------------------------------------
    print('Saving the final model to:', model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))

    print('Saving params to ', train_metadata_filepath)
    params = {'version': version,
              'n_estimators': n_estimators,
              'n_patients': n_train_patients,
              'samples': len(train_X),
              'total_time': duration_train,
              'scaler': scaler
              }
    results = {'params': params}
    with open(train_metadata_filepath, 'wb') as handle:
        pickle.dump(results, handle)
