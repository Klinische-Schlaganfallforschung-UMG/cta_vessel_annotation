"""
This script runs the whole pipeline including training, prediction and performance metrics evaluation
in k-fold cross-validation.

File: run_pipeline.py
Author: Jana Rieger
Created on: 24.08.2020
"""

import os
import pickle

from src.RF import rf_config
from src.RF.utils.evaluate_function import evaluate_and_save_to_csv
from src.RF.utils.predict_function import predict_and_save
from src.RF.utils.prepare_samples import get_balanced_samples
from src.RF.utils.train_function import train_and_save
from src.generalutils import helper
from sklearn import preprocessing

print('run_pipeline.py', os.getcwd())


def main():
    ################################################
    # SET PARAMETERS FOR TUNING
    ################################################
    n_estimators_list = [50, 100]  # list of number of trees in forest
    crossvalidation = False
    ################################################

    # general parameter
    version = rf_config.RF_VERSION
    models_dir = rf_config.MODELS_DIR
    data_dir = rf_config.DATA_DIR
    feature_names = rf_config.FEATURE_NAMES
    # additional parameters - predicting
    feature_filenames = rf_config.FEATURE_FILENAMES  # list of the feature inputs to the network in the right order
    predictions_dir = rf_config.PREDICTIONS_DIR
    # additional parameters - evaluation
    label_filename = rf_config.LABEL_FILENAMES[rf_config.H_LEVEL]
    class_dict = rf_config.CLASS_DICTS[rf_config.H_LEVEL]

    print('-' * 100)
    print('version', version)
    print('number of estimators list', n_estimators_list)
    print('crossvalidation', crossvalidation)
    print('*' * 100)

    # -----------------------------------------------------------
    # PREPARING MODEL SAMPLES
    # -----------------------------------------------------------
    if crossvalidation:
        # LOADING MODEL DATA
        X, y = get_balanced_samples(feature_names=feature_names)
        folds = helper.load_json(os.path.join('src', 'BRAVENET', 'xval_folds_example.json'))
        for f, fold in folds.items():
            if f not in ['0', '1', '2', '3']:
                continue
            print('FOLD %s' % f)
            datasets = [*fold]
            models_dir = os.path.join(rf_config.MODELS_DIR, 'fold' + str(f))
            predictions_dir = os.path.join(rf_config.PREDICTIONS_DIR, 'fold' + str(f))
            # Get train dfs.
            train_X = X[X['patient_id'].isin(fold['train']['working'])].drop('patient_id', axis=1)
            train_y = y[y['patient_id'].isin(fold['train']['working'])].drop('patient_id', axis=1)
            # Scaling
            scaler = preprocessing.MinMaxScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            # Patient numbers
            n_train_patients = len(fold['train']['working'])

            # -----------------------------------------------------------
            # RUNNING PIPELINE
            # -----------------------------------------------------------
            pipeline(n_estimators_list=n_estimators_list, train_X=train_X, train_y=train_y, scaler=scaler,
                     class_dict=class_dict, datasets=datasets, predictions_dir=predictions_dir, models_dir=models_dir,
                     data_dir=data_dir, version=version, feature_filenames=feature_filenames,
                     label_filename=label_filename, patients=fold, n_train_patients=n_train_patients)
    else:
        # LOADING MODEL DATA
        patients = rf_config.PATIENTS
        datasets = ['train', 'val', 'test']
        X, y = get_balanced_samples(feature_names=feature_names)
        # Get train dfs
        train_X = X[X['patient_id'].isin(patients['train']['working'])].drop('patient_id', axis=1)
        train_y = y[y['patient_id'].isin(patients['train']['working'])].drop('patient_id', axis=1)
        # Scaling
        scaler = preprocessing.MinMaxScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        # Patient numbers
        n_train_patients = len(patients['train']['working'])

        # -----------------------------------------------------------
        # RUNNING PIPELINE
        # -----------------------------------------------------------
        pipeline(n_estimators_list=n_estimators_list, train_X=train_X, train_y=train_y, scaler=scaler,
                 class_dict=class_dict, datasets=datasets, predictions_dir=predictions_dir, models_dir=models_dir,
                 data_dir=data_dir, version=version, feature_filenames=feature_filenames,
                 label_filename=label_filename, patients=patients, n_train_patients=n_train_patients)
    print('DONE')


def pipeline(n_estimators_list, train_X, train_y, scaler, class_dict, datasets, predictions_dir, models_dir, data_dir,
             version, feature_filenames, label_filename, patients, n_train_patients):
    # -----------------------------------------------------------
    # PREPARE FILES FOR STORING RESULTS
    # -----------------------------------------------------------
    all_classes = [*class_dict]
    all_classes = list(map(str, all_classes))
    result_files_dict = {}
    for dataset in datasets:
        if not os.path.exists(os.path.join(predictions_dir, dataset)):
            os.makedirs(os.path.join(predictions_dir, dataset))
        result_file = rf_config.get_complete_result_table_filepath(predictions_dir, dataset)
        result_file_per_patient = rf_config.get_complete_result_table_per_patient_filepath(predictions_dir, dataset)
        header_row = ['number of estimators',
                      'num pat ' + dataset, 'value',
                      'acc ' + dataset,
                      'dice macro ' + dataset,
                      'dice macro wo 0 ' + dataset,
                      'dice macro wo 0 and 1 ' + dataset,
                      'avg acc ' + dataset,
                      'avg acc wo 0 ' + dataset,
                      'avg acc wo 0 and 1 ' + dataset,
                      'dice binary ' + dataset] + all_classes
        header_row_patient = ['number of estimators',
                              'patient',
                              'acc ' + dataset,
                              'dice macro ' + dataset,
                              'dice macro wo 0 ' + dataset,
                              'dice macro wo 0 and 1 ' + dataset,
                              'avg acc ' + dataset,
                              'avg acc wo 0 ' + dataset,
                              'avg acc wo 0 and 1 ' + dataset,
                              'dice binary ' + dataset] + all_classes
        helper.write_to_csv(result_file, [header_row])
        helper.write_to_csv(result_file_per_patient, [header_row_patient])
        result_files_dict[dataset] = [result_file, result_file_per_patient]

    # -----------------------------------------------------------
    # FINE GRID FOR PARAMETER TUNING
    # -----------------------------------------------------------
    for n_estimators in n_estimators_list:
        run_name = rf_config.get_run_name(version=version, n_estimators=n_estimators,
                                          n_patients_train=n_train_patients)
        train_metadata_filepath = rf_config.get_train_metadata_filepath(models_dir, run_name)
        model_filepath = rf_config.get_model_filepath(models_dir, run_name)
        # -----------------------------------------------------------
        # TRAIN RF AND SAVE MODEL AND RESULTS
        # -----------------------------------------------------------
        train_and_save(version, models_dir, n_estimators=n_estimators,
                       n_train_patients=n_train_patients,
                       train_X=train_X, train_y=train_y,
                       scaler=scaler, run_name=run_name, train_metadata_filepath=train_metadata_filepath,
                       model_filepath=model_filepath)
        print('_' * 100)
        print('Training completed.')
        print('_' * 100)
        # -----------------------------------------------------------
        # PREDICT AND EVALUATE
        # -----------------------------------------------------------
        print('Starting prediction...')
        print('_' * 100)
        train_metadata = helper.load_model_metadata(train_metadata_filepath)
        model = pickle.load(open(model_filepath, 'rb'))
        for dataset in datasets:
            print('DATASET', dataset)
            set_patients = patients[dataset]['working']
            for patient in set_patients:
                print('-' * 100)
                predict_and_save(version=version, n_estimators=n_estimators,
                                 n_patients_train=n_train_patients, patient=patient,
                                 data_dir=data_dir, dataset=dataset,
                                 feature_filenames=feature_filenames,
                                 models_dir=models_dir, predictions_dir=predictions_dir, run_name=run_name,
                                 model=model, train_metadata=train_metadata)
            print('_' * 100)
            print('Prediction in %s dataset completed.' % dataset)
            print('_' * 100)
            # -----------------------------------------------------------
            # EVALUATE
            # -----------------------------------------------------------
            print('Starting evaluation in %s dataset...' % dataset)
            print('_' * 100)
            evaluate_and_save_to_csv(version=version, n_estimators=n_estimators,
                                     n_patients_train=n_train_patients, data_dir=data_dir,
                                     dataset=dataset,
                                     models_dir=models_dir, predictions_dir=predictions_dir,
                                     patients=set_patients, result_file=result_files_dict[dataset][0],
                                     result_file_per_patient=result_files_dict[dataset][1],
                                     label_load_name=label_filename, washed=False)
            print('_' * 100)
            print('Evaluation in %s dataset completed.' % dataset)
            print('_' * 100)


if __name__ == '__main__':
    main()
