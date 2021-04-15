"""
This script contains a function that calculates performance measures such as accuracy, average class accuracy and
DICE coefficient for given predictions and ground-truth labels and saves the results to a given csv file.
"""

import os
import pickle
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.RF import rf_config
from src.generalutils import helper
from src.generalutils.metrics import balanced_accuracy_score


def evaluate_and_save_to_csv(version, n_estimators,
                             n_patients_train, data_dir, dataset, models_dir, predictions_dir,
                             patients, result_file, result_file_per_patient, label_load_name,
                             washed):
    """

    :param version: The Unet version name. String.
    :param n_estimators: Number of trees in forest. Positive integer.
    :param n_patients_train: The number of patient used for training. Positive integer.
    :param data_dir: The path to data dirs. String.
    :param dataset: The dataset. E.g. train/val/set
    :param models_dir: Directory where trained models are saved. String.
    :param predictions_dir: Directory where predictions are saved. String.
    :param patients: The names of patients in the given dataset. List of strings.
    :param result_file: The path to the csv file where to store the results of the performance calculation calculated
    as average over all patient in dataset. String.
    :param result_file_per_patient: The path to the csv file where to store the results of the performance calculation
    per patient. String.
    :param label_load_name: The file name of the ground-truth label. String.
    :param washed: True for evaluation of predictions washed with vessel segments.
    :return:
    """
    print('model version', version)
    print('number of estimators', n_estimators)
    print('label load name', label_load_name)
    print('washed', washed)

    starttime_row = helper.start_time_measuring(what_to_measure='model evaluation')

    # create the name of current run
    run_name = rf_config.get_run_name(version=version, n_estimators=n_estimators, n_patients_train=n_patients_train)

    # -----------------------------------------------------------
    # TRAINING PARAMS
    # -----------------------------------------------------------
    try:
        train_metadata_filepath = rf_config.get_train_metadata_filepath(models_dir, run_name)
        with open(train_metadata_filepath, 'rb') as handle:
            train_metadata = pickle.load(handle)
        print('Train params:')
        print(train_metadata['params'])
    except FileNotFoundError:
        print('Unexpected error by reading train metadata:', sys.exc_info()[0])

    # -----------------------------------------------------------
    # DATASET RESULTS (TRAIN / VAL / TEST)
    # -----------------------------------------------------------
    # initialize empty list for saving measures for each patient
    acc_list = []
    f1_macro_list = []
    f1_macro_wo_0_list = []
    f1_macro_wo_0_and_1_list = []
    avg_acc_list = []
    avg_acc_wo_0_list = []
    avg_acc_wo_0_and_1_list = []
    f1_bin_list = []
    f1_class_dict = {}  # for saving f1 for each class
    for cls in range(rf_config.NUM_CLASSES):
        f1_class_dict[cls] = []

    # calculate the performance per patient
    for patient in patients:
        # load patient ground truth and prediction
        print('-----------------------------------------------------------------')
        print(patient)
        print('> Loading label...')
        label = helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + label_load_name))
        print('> Loading prediction...')
        if washed:
            prediction_path = rf_config.get_washed_annotation_filepath(predictions_dir, run_name, patient, dataset)
        else:
            prediction_path = rf_config.get_annotation_filepath(predictions_dir, run_name, patient, dataset)
        if not os.path.exists(prediction_path) and prediction_path.endswith('.gz'):
            prediction_path = prediction_path[:-3]
        prediction = helper.load_nifti_mat_from_file(prediction_path)

        # flatten the 3d volumes to 1d vector, necessary for performance calculation with sklearn functions
        label_f = label.flatten()
        prediction_f = prediction.flatten().astype(np.int16)
        label_f_bin = np.clip(label_f, 0, 1)
        prediction_f_bin = np.clip(prediction_f, 0, 1)

        print('Computing performance measures...')
        # classification accuracy
        acc = accuracy_score(label_f, prediction_f)
        # dice score for multiclass classification
        f1_per_classes = f1_score(label_f, prediction_f,
                                  average=None)  # f1 for each present class label in ground truth and prediction
        f1_macro = np.mean(f1_per_classes)  # averaged f1 over all present class labels
        f1_macro_wo_0 = np.mean(
            f1_per_classes[1:])  # averaged f1 over all present class labels except background class label
        f1_macro_wo_0_and_1 = np.mean(
            f1_per_classes[
            2:])  # averaged f1 over all present class labels except background  and not-annotated class label
        # average class accuracy for multiclass classification
        avg_acc, avg_acc_per_classes = balanced_accuracy_score(label_f, prediction_f)
        avg_acc_wo_0 = np.mean(avg_acc_per_classes[1:])
        avg_acc_wo_0_and_1 = np.mean(avg_acc_per_classes[2:])
        # dice for binary prediction -> labels converted to 1 for annotated vessels and 0 for background
        f1_bin = f1_score(label_f_bin, prediction_f_bin)
        patient_f1_class_list = ['-'] * rf_config.NUM_CLASSES  # for saving f1 for each class

        # print out to console
        print('acc:', acc)
        print('f1 all classes:', f1_macro)
        print('f1 without background class:', f1_macro_wo_0)
        print('f1 without background and not-annotated class:', f1_macro_wo_0_and_1)
        print('avg_acc all classes:', avg_acc)
        print('avg_acc without background class:', avg_acc_wo_0)
        print('avg_acc without background and not-annotated class:', avg_acc_wo_0_and_1)
        print('f1 binary predictions:', f1_bin)
        print('f1 per classes', f1_per_classes, 'size', len(f1_per_classes))

        # find what labels are in ground-truth and what labels were predicted
        unique_labels = np.unique(label_f)
        unique_prediction = np.unique(prediction_f)
        print('label unique', unique_labels, 'size', len(unique_labels))
        print('predicted annotation unique', unique_prediction, 'size', len(unique_prediction))

        # save results for patient to the lists
        acc_list.append(acc)
        f1_macro_list.append(f1_macro)
        f1_macro_wo_0_list.append(f1_macro_wo_0)
        f1_macro_wo_0_and_1_list.append(f1_macro_wo_0_and_1)
        avg_acc_list.append(avg_acc)
        avg_acc_wo_0_list.append(avg_acc_wo_0)
        avg_acc_wo_0_and_1_list.append(avg_acc_wo_0_and_1)
        f1_bin_list.append(f1_bin)
        all_classes = np.concatenate((unique_labels, unique_prediction))
        unique_classes = np.unique(all_classes)
        for i, cls in enumerate(unique_classes):
            f1_class_dict[cls].append(f1_per_classes[i])
            patient_f1_class_list[cls] = f1_per_classes[i]

        # create row for saving to csv file with details for each patient and write to the csv file
        row_per_patient = [n_estimators, patient,
                           acc,
                           f1_macro,
                           f1_macro_wo_0,
                           f1_macro_wo_0_and_1,
                           avg_acc,
                           avg_acc_wo_0,
                           avg_acc_wo_0_and_1,
                           f1_bin] + patient_f1_class_list
        helper.write_to_csv(result_file_per_patient, [row_per_patient])

    # calculate patient mean and std for each class
    f1_class_list_mean = []
    f1_class_list_std = []
    for cls, class_f1_list in f1_class_dict.items():
        f1_class_list_mean.append(np.mean(class_f1_list))
        f1_class_list_std.append(np.std(class_f1_list))
    # create row for saving to csv file with averages over whole set and write to the csv file
    row_avg = [n_estimators, len(patients),
               'AVG',
               np.mean(acc_list),
               np.mean(f1_macro_list),
               np.mean(f1_macro_wo_0_list),
               np.mean(f1_macro_wo_0_and_1_list),
               np.mean(avg_acc_list),
               np.mean(avg_acc_wo_0_list),
               np.mean(avg_acc_wo_0_and_1_list),
               np.mean(f1_bin_list)] + f1_class_list_mean
    row_std = [n_estimators, len(patients),
               'STD',
               np.std(acc_list),
               np.std(f1_macro_list),
               np.std(f1_macro_wo_0_list),
               np.std(f1_macro_wo_0_and_1_list),
               np.std(avg_acc_list),
               np.std(avg_acc_wo_0_list),
               np.std(avg_acc_wo_0_and_1_list),
               np.std(f1_bin_list)] + f1_class_list_std
    print('AVG:', row_avg)
    print('STD:', row_std)
    helper.write_to_csv(result_file, [row_avg, row_std])

    # print out how long did the calculations take
    helper.end_time_measuring(starttime_row, what_to_measure='model evaluation')
