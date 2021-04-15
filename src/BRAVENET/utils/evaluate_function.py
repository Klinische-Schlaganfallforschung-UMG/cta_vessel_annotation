"""
This script contains a function that calculates performance measures such as accuracy, average class accuracy and
DICE coefficient for given predictions and ground-truth labels and saves the results to a given csv file.
"""

import os
import pickle
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from termcolor import colored

from src.BRAVENET import bravenet_config
from src.generalutils import helper
from src.generalutils.metrics import balanced_accuracy_score


def evaluate_set(version, n_epochs, batch_size, learning_rate, dropout_rate, l1, l2, batch_normalization,
                 deconvolution, n_base_filters, n_patients_train, n_patients_val, patients, data_dir,
                 predictions_dir, models_dir, dataset, result_file, result_file_per_patient, label_load_name,
                 washed=False, washing_version='score', mode='voxel-wise', segment_load_name=''):
    """
    :param version: The net version name. String.
    :param n_epochs: The number of epochs. Positive integer.
    :param batch_size: The size of one mini-batch. Positive integer.
    :param learning_rate: The learning rate. Positive float.
    :param dropout_rate: The dropout rate. Positive float or None.
    :param l1: The L1 regularization. Positive float or None.
    :param l2: The L2 regularization. Positive float or None.
    :param batch_normalization: Whether to train with batch normalization. Boolean.
    :param deconvolution: Whether to use deconvolution instead of up-sampling layer. Boolean.
    :param n_base_filters: The number of filters in the first convolutional layer of the net. Positive integer.
    :param n_patients_train: The number of patient used for training. Positive integer.
    :param n_patients_val: The number of patient used for validation. Positive integer.
    :param patients: The names of patients in the given dataset. List of strings.
    :param data_dir: The path to data dir. String.
    :param predictions_dir: Path to directory where to save predictions and results. String.
    :param models_dir: Path to directory where to save models. String.
    :param dataset: The dataset. E.g. train/val/set
    :param result_file: The path to the csv file where to store the results of the performance calculation calculated
    as average over all patient in dataset. String.
    :param result_file_per_patient: The path to the csv file where to store the results of the performance calculation
    per patient. String.
    :param label_load_name: The file name of the ground-truth label. String.
    :param washed: True for washed predictions with vessels segments.
    :param washing_version: The name of the washing version. Can be empty. String.
    :param mode: One of the values ['voxel-wise', 'segment-wise']. Voxel-wise: voxel-wise scores are calculated.
    Segment-wise: segment-wise scores are calculated. String.
    :param segment_load_name: Filename regex for files containing the skeletons with vessel segments.
    :return:
    """
    print('label load name', label_load_name)
    print('washed', washed)
    print('mode', mode)

    starttime_set = helper.start_time_measuring('performance assessment in set')

    # create the name of current run
    run_name = bravenet_config.get_run_name(version=version, n_epochs=n_epochs, batch_size=batch_size,
                                            learning_rate=learning_rate, dropout_rate=dropout_rate, l1=l1, l2=l2,
                                            batch_normalization=batch_normalization, deconvolution=deconvolution,
                                            n_base_filters=n_base_filters,
                                            n_patients_train=n_patients_train, n_patients_val=n_patients_val)

    # -----------------------------------------------------------
    # TRAINING PARAMS
    # -----------------------------------------------------------
    try:
        train_metadata_filepath = bravenet_config.get_train_metadata_filepath(models_dir, run_name)
        with open(train_metadata_filepath, 'rb') as handle:
            train_metadata = pickle.load(handle)
        print('Train params:', train_metadata['params'])
    except FileNotFoundError:
        train_metadata = None
        print('Unexpected error by reading train metadata:', sys.exc_info()[0])

    # -----------------------------------------------------------
    # DATASET RESULTS (TRAIN / VAL / TEST)
    # -----------------------------------------------------------
    num_epochs = train_metadata['params']['epochs'] if train_metadata else n_epochs
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
    for cls in range(bravenet_config.NUM_CLASSES):
        f1_class_dict[cls] = []

    # calculate the performance per patient
    for p, patient in enumerate(patients):
        # load patient ground truth and prediction
        print('-----------------------------------------------------------------')
        print('PATIENT:', patient, ',', str(p + 1) + '/' + str(len(patients)))
        print('> Loading label...')
        label = helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + label_load_name))
        print('> Loading prediction...')
        if washed:
            prediction_path = bravenet_config.get_washed_annotation_filepath(predictions_dir, run_name, patient,
                                                                             dataset, washing_version=washing_version)
        else:
            prediction_path = bravenet_config.get_annotation_filepath(predictions_dir, run_name, patient,
                                                                      dataset)
        if not os.path.exists(prediction_path) and prediction_path.endswith('.gz'):
            prediction_path = prediction_path[:-3]
        prediction = helper.load_nifti_mat_from_file(prediction_path)

        # If mode is segment-wise, prepare the segment-wise data points.
        if mode == 'segment-wise':
            print('> Loading segments...')
            skel_seg = helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + segment_load_name))

            # Get label and prediction segment datapoints.
            label_segment_datapoints = []
            prediction_segment_datapoints = []
            unique_segments = np.unique(skel_seg)
            for seg in unique_segments:
                segment_inds = np.where(skel_seg == seg)

                # Get label segment data points.
                label_segment_classes = label[segment_inds]
                label_segment_unique_classes, label_segment_classes_counts = np.unique(label_segment_classes,
                                                                                       return_counts=True)
                if len(label_segment_unique_classes) > 1:
                    label_segment_class = label_segment_unique_classes[np.argmax(label_segment_classes_counts)]
                else:
                    label_segment_class = label_segment_unique_classes[0]
                label_segment_datapoints.append(label_segment_class)

                # Get prediction segment data points.
                prediction_segment_classes = prediction[segment_inds]
                prediction_segment_unique_classes, prediction_segment_classes_counts = np.unique(
                    prediction_segment_classes,
                    return_counts=True)
                if len(prediction_segment_unique_classes) > 1:
                    prediction_segment_class = prediction_segment_unique_classes[
                        np.argmax(prediction_segment_classes_counts)]
                else:
                    prediction_segment_class = prediction_segment_unique_classes[0]
                prediction_segment_datapoints.append(prediction_segment_class)

            # assign the segment datapoint lists to flatten variables
            label_f = label_segment_datapoints
            prediction_f = prediction_segment_datapoints
        elif mode == 'voxel-wise':
            # flatten the 3d volumes to 1d vector, necessary for performance calculation with sklearn functions
            label_f = label.flatten()
            prediction_f = prediction.flatten()
        else:
            raise ValueError('Mode can be only one of the values ["voxel-wise", "segment-wise"].')

        # Calculate scores.
        scores = evaluate_volume(label_f, prediction_f)

        # find what labels are in ground-truth and what labels were predicted
        unique_labels = np.unique(label_f)
        unique_prediction = np.unique(prediction_f)
        print('label unique', unique_labels, 'size', len(unique_labels))
        print('predicted annotation unique', unique_prediction, 'size', len(unique_prediction))

        # save results for patient to the lists
        acc_list.append(scores['acc'])
        f1_macro_list.append(scores['f1_macro'])
        f1_macro_wo_0_list.append(scores['f1_macro_wo_0'])
        f1_macro_wo_0_and_1_list.append(scores['f1_macro_wo_0_and_1'])
        avg_acc_list.append(scores['avg_acc'])
        avg_acc_wo_0_list.append(scores['avg_acc_wo_0'])
        avg_acc_wo_0_and_1_list.append(scores['avg_acc_wo_0_and_1'])
        f1_bin_list.append(scores['f1_bin'])
        all_classes = np.concatenate((unique_labels, unique_prediction))
        unique_classes = np.unique(all_classes)
        f1_per_classes = scores['f1_per_classes']
        patient_f1_class_list = ['-'] * bravenet_config.NUM_CLASSES  # for saving f1 for each class
        for i, cls in enumerate(unique_classes):
            f1_class_dict[cls].append(f1_per_classes[i])
            patient_f1_class_list[cls] = f1_per_classes[i]

        # create row for saving to csv file with details for each patient and write to the csv file
        row_per_patient = [num_epochs, batch_size, learning_rate, dropout_rate, l1, l2, batch_normalization,
                           deconvolution, n_base_filters, patient,
                           scores['acc'],
                           scores['f1_macro'],
                           scores['f1_macro_wo_0'],
                           scores['f1_macro_wo_0_and_1'],
                           scores['avg_acc'],
                           scores['avg_acc_wo_0'],
                           scores['avg_acc_wo_0_and_1'],
                           scores['f1_bin']] + patient_f1_class_list
        print('Writing to per patient csv...')
        helper.write_to_csv(result_file_per_patient, [row_per_patient])

    # calculate patient mean and std for each class
    f1_class_list_mean = []
    f1_class_list_std = []
    for cls, class_f1_list in f1_class_dict.items():
        f1_class_list_mean.append(np.mean(class_f1_list))
        f1_class_list_std.append(np.std(class_f1_list))
    # create row for saving to csv file with averages over whole set and write to the csv file
    row_avg = [num_epochs, batch_size, learning_rate, dropout_rate, l1, l2, batch_normalization, deconvolution,
               n_base_filters, len(patients),
               'AVG',
               np.mean(acc_list),
               np.mean(f1_macro_list),
               np.mean(f1_macro_wo_0_list),
               np.mean(f1_macro_wo_0_and_1_list),
               np.mean(avg_acc_list),
               np.mean(avg_acc_wo_0_list),
               np.mean(avg_acc_wo_0_and_1_list),
               np.mean(f1_bin_list)] + f1_class_list_mean
    row_std = [num_epochs, batch_size, learning_rate, dropout_rate, l1, l2, batch_normalization, deconvolution,
               n_base_filters, len(patients),
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
    print('Writing to csv...')
    helper.write_to_csv(result_file, [row_avg, row_std])

    # print out how long did the calculations take
    helper.end_time_measuring(starttime_set, what_to_measure='performance assessment in set')


def evaluate_volume(label_f, prediction_f):
    # binary values for calculating binary scores
    label_f_bin = np.clip(label_f, 0, 1)
    prediction_f_bin = np.clip(prediction_f, 0, 1)

    print('Computing performance measures...')
    # classification accuracy
    acc = accuracy_score(label_f, prediction_f)
    # dice score for multiclass classification
    f1_per_classes = f1_score(label_f, prediction_f,
                              average=None)  # f1 for each present class label in ground truth and prediction
    f1_macro = np.mean(f1_per_classes)  # averaged f1 over all present class labels
    f1_macro_wo_0 = np.mean(f1_per_classes[1:])  # averaged f1 over all present class labels except background
    f1_macro_wo_0_and_1 = np.mean(
        f1_per_classes[2:])  # averaged f1 over all present class labels except background and not-annotated
    # average class accuracy multiclass classification
    avg_acc, avg_acc_per_classes = balanced_accuracy_score(label_f, prediction_f)
    avg_acc_wo_0 = np.mean(avg_acc_per_classes[1:])
    avg_acc_wo_0_and_1 = np.mean(avg_acc_per_classes[2:])
    # dice for binary prediction -> labels converted to 1 for annotated vessels and 0 for background
    f1_bin = f1_score(label_f_bin, prediction_f_bin)

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

    scores = {'acc': acc,
              'f1_per_classes': f1_per_classes,
              'f1_macro': f1_macro,
              'f1_macro_wo_0': f1_macro_wo_0,
              'f1_macro_wo_0_and_1': f1_macro_wo_0_and_1,
              'avg_acc': avg_acc,
              'avg_acc_wo_0': avg_acc_wo_0,
              'avg_acc_wo_0_and_1': avg_acc_wo_0_and_1,
              'f1_bin': f1_bin}

    return scores


def get_csv_headers(dataset, all_classes):
    header_row = ['num epochs', 'batch size', 'learning rate', 'dropout rate', 'l1', 'l2', 'batch normalization',
                  'deconvolution', 'number of base filters', 'num pat ' + dataset, 'value',
                  'acc ' + dataset,
                  'dice macro ' + dataset,
                  'dice macro wo 0 ' + dataset,
                  'dice macro wo 0 and 1 ' + dataset,
                  'avg acc ' + dataset,
                  'avg acc wo 0 ' + dataset,
                  'avg acc wo 0 and 1 ' + dataset,
                  'dice binary ' + dataset] + all_classes
    header_row_patient = ['num epochs', 'batch size', 'learning rate', 'dropout rate', 'l1', 'l2',
                          'batch normalization', 'deconvolution', 'number of base filters', 'patient',
                          'acc ' + dataset,
                          'dice macro ' + dataset,
                          'dice macro wo 0 ' + dataset,
                          'dice macro wo 0 and 1 ' + dataset,
                          'avg acc ' + dataset,
                          'avg acc wo 0 ' + dataset,
                          'avg acc wo 0 and 1 ' + dataset,
                          'dice binary ' + dataset] + all_classes
    return header_row, header_row_patient


def main(label_filepath, prediction_filepath, mode, segments_filepath=''):
    print('> Loading label...')
    label = helper.load_nifti_mat_from_file(label_filepath)
    print('> Loading prediction...')
    prediction = helper.load_nifti_mat_from_file(prediction_filepath)

    # If mode is segment-wise, prepare the segment-wise data points.
    if mode == 'segment-wise':
        print('> Loading segments...')
        skel_seg = helper.load_nifti_mat_from_file(segments_filepath)

        # Get label and prediction segment datapoints.
        label_segment_datapoints = []
        prediction_segment_datapoints = []
        unique_segments = np.unique(skel_seg)
        for seg in unique_segments:
            segment_inds = np.where(skel_seg == seg)

            # Get label segment data points.
            label_segment_classes = label[segment_inds]
            label_segment_unique_classes, label_segment_classes_counts = np.unique(label_segment_classes,
                                                                                   return_counts=True)
            if len(label_segment_unique_classes) > 1:
                print(colored(('Label - Segment id:', seg,
                               ' Unique classes:', label_segment_unique_classes,
                               ' Classes counts: ', label_segment_classes_counts), 'red'))
                label_segment_class = label_segment_unique_classes[np.argmax(label_segment_classes_counts)]
            else:
                label_segment_class = label_segment_unique_classes[0]
            label_segment_datapoints.append(label_segment_class)

            # Get prediction segment data points.
            prediction_segment_classes = prediction[segment_inds]
            prediction_segment_unique_classes, prediction_segment_classes_counts = np.unique(
                prediction_segment_classes,
                return_counts=True)
            if len(prediction_segment_unique_classes) > 1:
                print(colored(('Prediction - Segment id:', seg,
                               ' Unique classes:', prediction_segment_unique_classes,
                               ' Classes counts: ', prediction_segment_classes_counts), 'red'))
                prediction_segment_class = prediction_segment_unique_classes[
                    np.argmax(prediction_segment_classes_counts)]
            else:
                prediction_segment_class = prediction_segment_unique_classes[0]
            prediction_segment_datapoints.append(prediction_segment_class)

        # assign the segment datapoint lists to flatten variables
        label_f = label_segment_datapoints
        prediction_f = prediction_segment_datapoints
    elif mode == 'voxel-wise':
        # flatten the 3d volumes to 1d vector, necessary for performance calculation with sklearn functions
        label_f = label.flatten()
        prediction_f = prediction.flatten()
    else:
        raise ValueError('Mode can be only one of the values ["voxel-wise", "segment-wise"].')

    scores = evaluate_volume(label_f, prediction_f)

    return scores


if __name__ == '__main__':
    main(label_filepath=sys.argv[1], prediction_filepath=sys.argv[2], mode=sys.argv[3])
