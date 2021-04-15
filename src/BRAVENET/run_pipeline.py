"""
This script runs the whole pipeline including training, prediction and performance metrics evaluation.
It can be used for hyperparameter tuning on fine grid and also for cross-validation.

File: run_pipeline.py
Author: Jana Rieger
Created on: 18.05.2020
"""

import os

import numpy as np
from keras import backend as K
from tensorflow.keras.models import load_model

from src.BRAVENET import bravenet_config
from src.BRAVENET.utils.evaluate_function import evaluate_set, get_csv_headers
from src.BRAVENET.utils.helper_model_samples import train_feature_label_lists, val_feature_label_lists
from src.BRAVENET.utils.predict_function import predict
from src.BRAVENET.utils.train_function import train
from src.generalutils import helper

print('run_pipeline.py', os.getcwd())


def main():
    ################################################
    # SET PARAMETERS
    ################################################
    # grid search for training
    nr_epochs = 4  # number of epochs
    select_min_loss = True
    batch_size_list = [2]  # list with batch sizes
    learning_rate_list = [1e-3]  # list with learning rates of the optimizer Adam
    dropout_rate_list = [None]  # set None if no dropout
    l1_list = [None]  # set None if no L1 regularization
    l2_list = [1e-3]  # set None if no L2 regularization
    batch_normalization_list = [True]  # whether to use batch normalization after each conv layer
    deconvolution_list = [False]  # whether to use deconvolution instead of up-sampling layer
    nr_base_filters_list = [16]  # number of filter in the first convolutional layer
    depth = 4  # number of levels of the model
    filter_size = (3, 3, 3)  # size of the convolutional filters, just for 3D version
    checkpoint_model = True  # whether to train with keras ModelCheckpoint callback
    normalize_features = True
    augmentation = True
    crossvalidation = True
    # additional for predicting, not for crossvalidation
    datasets = ['val', 'train']  # train / val / test
    # additional for evaluation
    washing_version = ''
    ################################################
    # general parameters - training
    data_dir = bravenet_config.DATA_DIR
    version = bravenet_config.NET_VERSION
    transfer_learning = 'transfer' in version
    transfer_weights_path = os.path.join(bravenet_config.TOP_LEVEL_DATA, 'brainseg_weights/weights.h5')
    num_classes = bravenet_config.NUM_CLASSES
    activation = bravenet_config.ACTIVATION
    final_activation = bravenet_config.FINAL_ACTIVATION
    loss_function = bravenet_config.LOSS_FUNCTION
    metrics = bravenet_config.METRICS
    optimizer = bravenet_config.OPTIMIZER
    patch_size = (bravenet_config.PATCH_SIZE_X, bravenet_config.PATCH_SIZE_Y, bravenet_config.PATCH_SIZE_Z)
    # additional parameters - predicting
    feature_file_names = bravenet_config.FEATURE_FILENAMES  # list of the feature inputs to the network in the right order
    models_dir = bravenet_config.MODELS_DIR
    predictions_dir = bravenet_config.PREDICTIONS_DIR
    patch_size_x = bravenet_config.PATCH_SIZE_X
    patch_size_y = bravenet_config.PATCH_SIZE_Y
    patch_size_z = bravenet_config.PATCH_SIZE_Z
    # additional parameters - evaluation
    label_file_name = bravenet_config.LABEL_FILENAMES[bravenet_config.H_LEVEL]
    class_dict = bravenet_config.CLASS_DICTS[bravenet_config.H_LEVEL]

    print('-' * 100)
    print('version', version)
    print('transfer learning', transfer_learning)
    print('loss function', loss_function)
    print('metric', metrics)
    print('patch size', patch_size)
    print('number of epochs', nr_epochs)
    print('batch size list', batch_size_list)
    print('learning rate list', learning_rate_list)
    print('dropout rate list', dropout_rate_list)
    print('L1 regularization list', l1_list)
    print('L2 regularization list', l2_list)
    print('batch normalization list', batch_normalization_list)
    print('deconvolution list', deconvolution_list)
    print('number of base filters', nr_base_filters_list)
    print('depth', depth)
    print('filter size', filter_size)
    print('checkpoint model', checkpoint_model)
    print('normalize features', normalize_features)
    print('augmentation', augmentation)
    print('patch path', bravenet_config.SAMPLES_PATH)
    print('crossvalidation', crossvalidation)
    print('*' * 100)

    # -----------------------------------------------------------
    # PREPARING MODEL SAMPLES
    # -----------------------------------------------------------
    if crossvalidation:
        folds = helper.load_json(os.path.join('src', 'BRAVENET', 'xval_folds_example.json'))
        for f, fold in folds.items():
            if f not in ['0', '1', '2', '3']:
                continue
            print('FOLD %s' % f)
            datasets = [*fold]
            models_dir = os.path.join(bravenet_config.MODELS_DIR, 'fold' + str(f))
            predictions_dir = os.path.join(bravenet_config.PREDICTIONS_DIR, 'fold' + str(f))
            # Lists of patient
            listdir = os.listdir(bravenet_config.SAMPLES_PATH)
            train_feature_files, train_label_files, train_feature_files_big, train_label_files_big = train_feature_label_lists(
                listdir, patch_size, fold['train']['working'], augmentation)
            val_feature_files, val_label_files, val_feature_files_big, val_label_files_big = val_feature_label_lists(
                listdir, patch_size, fold['test']['working'])
            # Patient numbers
            if augmentation:
                n_train_patients = helper.number_of_patients('train', train_feature_files_big, train_label_files_big)
            else:
                n_train_patients = helper.number_of_patients('train', train_feature_files, train_label_files)
            n_val_patients = helper.number_of_patients('val', val_feature_files, val_label_files)
            assert n_train_patients == len(fold['train']['working'])
            assert n_val_patients == len(fold['test']['working'])
            # values for normalization
            max_values_for_normalization = get_max_values(fold['train']['working'], data_dir, feature_file_names)

            # -----------------------------------------------------------
            # RUNNING PIPELINE
            # -----------------------------------------------------------
            pipeline(class_dict=class_dict, datasets=datasets, predictions_dir=predictions_dir, models_dir=models_dir,
                     data_dir=data_dir, batch_size_list=batch_size_list, learning_rate_list=learning_rate_list,
                     dropout_rate_list=dropout_rate_list, l1_list=l1_list, l2_list=l2_list,
                     batch_normalization_list=batch_normalization_list, deconvolution_list=deconvolution_list,
                     nr_base_filters_list=nr_base_filters_list, version=version, nr_epochs=nr_epochs, depth=depth,
                     filter_size=filter_size, activation=activation, final_activation=final_activation,
                     num_classes=num_classes, optimizer=optimizer, loss_function=loss_function, metrics=metrics,
                     n_train_patients=n_train_patients, n_val_patients=n_val_patients,
                     checkpoint_model=checkpoint_model, train_feature_files=train_feature_files,
                     train_feature_files_big=train_feature_files_big, train_label_files=train_label_files,
                     train_label_files_big=train_label_files_big, val_feature_files=val_feature_files,
                     val_feature_files_big=val_feature_files_big, val_label_files=val_label_files,
                     val_label_files_big=val_label_files_big, normalize_features=normalize_features,
                     max_values_for_normalization=max_values_for_normalization, transfer_learning=transfer_learning,
                     transfer_weights_path=transfer_weights_path, feature_file_names=feature_file_names,
                     patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_size_z=patch_size_z,
                     label_file_name=label_file_name, washing_version=washing_version, patients=fold,
                     select_min_loss=select_min_loss)
    else:
        patients = bravenet_config.PATIENTS
        max_values_for_normalization = bravenet_config.MAX_FEATURE_VALUES
        # Lists of model samples
        listdir = os.listdir(bravenet_config.SAMPLES_PATH)
        train_feature_files, train_label_files, train_feature_files_big, train_label_files_big = train_feature_label_lists(
            listdir, patch_size, patients['train']['working'], augmentation)
        val_feature_files, val_label_files, val_feature_files_big, val_label_files_big = val_feature_label_lists(
            listdir, patch_size, patients['val']['working'])
        # Patient numbers
        if augmentation:
            n_train_patients = helper.number_of_patients('train', train_feature_files_big, train_label_files_big)
        else:
            n_train_patients = helper.number_of_patients('train', train_feature_files, train_label_files)
        n_val_patients = helper.number_of_patients('val', val_feature_files, val_label_files)
        assert n_train_patients == len(patients['train']['working'])
        assert n_val_patients == len(patients['val']['working'])

        # -----------------------------------------------------------
        # RUNNING PIPELINE
        # -----------------------------------------------------------
        pipeline(class_dict=class_dict, datasets=datasets, predictions_dir=predictions_dir, models_dir=models_dir,
                 data_dir=data_dir, batch_size_list=batch_size_list, learning_rate_list=learning_rate_list,
                 dropout_rate_list=dropout_rate_list, l1_list=l1_list, l2_list=l2_list,
                 batch_normalization_list=batch_normalization_list, deconvolution_list=deconvolution_list,
                 nr_base_filters_list=nr_base_filters_list, version=version, nr_epochs=nr_epochs, depth=depth,
                 filter_size=filter_size, activation=activation, final_activation=final_activation,
                 num_classes=num_classes, optimizer=optimizer, loss_function=loss_function, metrics=metrics,
                 n_train_patients=n_train_patients, n_val_patients=n_val_patients, checkpoint_model=checkpoint_model,
                 train_feature_files=train_feature_files, train_feature_files_big=train_feature_files_big,
                 train_label_files=train_label_files, train_label_files_big=train_label_files_big,
                 val_feature_files=val_feature_files, val_feature_files_big=val_feature_files_big,
                 val_label_files=val_label_files, val_label_files_big=val_label_files_big,
                 normalize_features=normalize_features, max_values_for_normalization=max_values_for_normalization,
                 transfer_learning=transfer_learning, transfer_weights_path=transfer_weights_path,
                 feature_file_names=feature_file_names, patch_size_x=patch_size_x, patch_size_y=patch_size_y,
                 patch_size_z=patch_size_z, label_file_name=label_file_name, washing_version=washing_version,
                 patients=patients, select_min_loss=select_min_loss)
    print('DONE')


def pipeline(class_dict, datasets, predictions_dir, models_dir, data_dir, batch_size_list, learning_rate_list,
             dropout_rate_list, l1_list, l2_list, batch_normalization_list, deconvolution_list, nr_base_filters_list,
             version, nr_epochs, depth, filter_size, activation, final_activation, num_classes, optimizer,
             loss_function, metrics, n_train_patients, n_val_patients, checkpoint_model, train_feature_files,
             train_feature_files_big, train_label_files, train_label_files_big, val_feature_files,
             val_feature_files_big, val_label_files, val_label_files_big, normalize_features,
             max_values_for_normalization, transfer_learning, transfer_weights_path, feature_file_names,
             patch_size_x, patch_size_y, patch_size_z, label_file_name,
             washing_version, patients, select_min_loss):
    # -----------------------------------------------------------
    # PREPARE FILES FOR STORING RESULTS
    # -----------------------------------------------------------
    all_classes = [*class_dict]
    all_classes = list(map(str, all_classes))
    result_files_dict = {}
    result_files_dict_washed = {}
    result_files_dict_segment_wise = {}
    for dataset in datasets:
        if not os.path.exists(os.path.join(predictions_dir, dataset)):
            os.makedirs(os.path.join(predictions_dir, dataset))
        result_file = bravenet_config.get_complete_result_table_filepath(predictions_dir, dataset)
        result_file_per_patient = bravenet_config.get_complete_result_table_per_patient_filepath(predictions_dir, dataset)
        result_file_washed = bravenet_config.get_complete_result_table_filepath(predictions_dir, dataset, washed=True)
        result_file_per_patient_washed = bravenet_config.get_complete_result_table_per_patient_filepath(predictions_dir,
                                                                                                        dataset,
                                                                                                        washed=True)
        result_file_segment_wise = bravenet_config.get_complete_result_table_filepath(predictions_dir, dataset,
                                                                                      washed=True, segment_wise=True)
        result_file_per_patient_segment_wise = bravenet_config.get_complete_result_table_per_patient_filepath(
            predictions_dir, dataset, washed=True, segment_wise=True)
        header_row, header_row_patient = get_csv_headers(dataset, all_classes)
        helper.write_to_csv(result_file, [header_row])
        helper.write_to_csv(result_file_per_patient, [header_row_patient])
        helper.write_to_csv(result_file_washed, [header_row])
        helper.write_to_csv(result_file_per_patient_washed, [header_row_patient])
        helper.write_to_csv(result_file_segment_wise, [header_row])
        helper.write_to_csv(result_file_per_patient_segment_wise, [header_row_patient])
        result_files_dict[dataset] = [result_file, result_file_per_patient]
        result_files_dict_washed[dataset] = [result_file_washed, result_file_per_patient_washed]
        result_files_dict_segment_wise[dataset] = [result_file_segment_wise, result_file_per_patient_segment_wise]

    # -----------------------------------------------------------
    # FINE GRID FOR PARAMETER TUNING
    # -----------------------------------------------------------
    hyperparam_grid = []
    for bs in batch_size_list:
        for lr in learning_rate_list:
            for dr in dropout_rate_list:
                for l1 in l1_list:
                    for l2 in l2_list:
                        for bn in batch_normalization_list:
                            for deconv in deconvolution_list:
                                for nr_base_filt in nr_base_filters_list:
                                    hyperparam_grid.append((bs, lr, dr, l1, l2, bn, deconv, nr_base_filt))
    for bs, lr, dr, l1, l2, bn, deconv, nr_base_filt in hyperparam_grid:
        # -----------------------------------------------------------
        # TRAIN BRAVENET AND SAVE MODEL AND RESULTS
        # -----------------------------------------------------------
        train(version, n_epochs=nr_epochs, batch_size=bs, lr=lr, dr=dr, l1=l1, l2=l2, bn=bn,
              n_base_filters=nr_base_filt, depth=depth, filter_size=filter_size, activation=activation,
              final_activation=final_activation, n_classes=num_classes, optimizer=optimizer,
              loss_function=loss_function, metrics=metrics, deconvolution=deconv,
              n_train_patients=n_train_patients, n_val_patients=n_val_patients,
              checkpoint_model=checkpoint_model, models_dir=models_dir,
              train_feature_files=train_feature_files, train_feature_files_big=train_feature_files_big,
              train_label_files=train_label_files, train_label_files_big=train_label_files_big,
              val_feature_files=val_feature_files, val_feature_files_big=val_feature_files_big,
              val_label_files=val_label_files, val_label_files_big=val_label_files_big,
              normalize_features=normalize_features, max_values_for_normalization=max_values_for_normalization,
              transfer_learning=transfer_learning, transfer_weights_path=transfer_weights_path)
        print('_' * 100)
        print('Training completed.')
        print('_' * 100)
        # -----------------------------------------------------------
        # PREDICT AND EVALUATE
        # -----------------------------------------------------------
        print('Starting prediction...')
        print('_' * 100)
        run_name = bravenet_config.get_run_name(version=version, n_epochs=nr_epochs, batch_size=bs, learning_rate=lr,
                                                dropout_rate=dr, l1=l1, l2=l2, batch_normalization=bn,
                                                deconvolution=deconv, n_base_filters=nr_base_filt,
                                                n_patients_train=n_train_patients, n_patients_val=n_val_patients)
        train_metadata_filepath = bravenet_config.get_train_metadata_filepath(models_dir, run_name)
        train_metadata = helper.load_model_metadata(train_metadata_filepath)
        # Select model with min val loss from last 3 epochs.
        print('Last 3 epochs:', train_metadata['history']['val_loss'][-3:])
        if select_min_loss:
            selected_epoch = np.argmin(train_metadata['history']['val_loss'][-3:]) + nr_epochs - 2
        else:
            selected_epoch = nr_epochs
        print('Selected epoch:', selected_epoch)
        run_name = bravenet_config.get_run_name(version=version, n_epochs=selected_epoch, batch_size=bs,
                                                learning_rate=lr, dropout_rate=dr, l1=l1, l2=l2, batch_normalization=bn,
                                                deconvolution=deconv, n_base_filters=nr_base_filt,
                                                n_patients_train=n_train_patients, n_patients_val=n_val_patients)
        model, _ = load_model_and_metadata(run_name=run_name, models_dir=models_dir)
        for dataset in datasets:
            print('DATASET', dataset)
            set_patients = patients[dataset]['working']
            if not os.path.exists(os.path.join(predictions_dir, dataset)):
                os.makedirs(os.path.join(predictions_dir, dataset))
            for patient in set_patients:
                print('-' * 100)
                feature_volumes = helper.load_feature_volumes(feature_file_names=feature_file_names, data_dir=data_dir,
                                                              patient=patient,
                                                              num_features_for_check=bravenet_config.NUM_FEATURES)
                print('> Loading segmented skeleton...')
                skeleton_segments_path = os.path.join(data_dir, patient + '_skel_seg_y.nii.gz')
                skeleton_segments = helper.load_nifti_mat_from_file(skeleton_segments_path)
                washed_annotation_save_path = bravenet_config.get_washed_annotation_filepath(
                    predictions_dir=predictions_dir, run_name=run_name, patient=patient, dataset=dataset)
                annotation_save_path = bravenet_config.get_annotation_filepath(predictions_dir=predictions_dir,
                                                                               run_name=run_name, patient=patient,
                                                                               dataset=dataset)
                predict(feature_volumes=feature_volumes, model=model, num_classes=num_classes,
                        patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_size_z=patch_size_z,
                        annotation_save_path=annotation_save_path,
                        max_values_for_normalization=max_values_for_normalization, skeleton_segments=skeleton_segments,
                        washed_annotation_save_path=washed_annotation_save_path)
            print('_' * 100)
            print('Prediction in %s dataset completed.' % dataset)
            print('_' * 100)
            # -----------------------------------------------------------
            # EVALUATE
            # -----------------------------------------------------------
            print('Starting evaluation in %s dataset...' % dataset)
            print('_' * 100)
            evaluate_set(version=version, n_epochs=selected_epoch, batch_size=bs, learning_rate=lr, dropout_rate=dr,
                         l1=l1,
                         l2=l2, batch_normalization=bn, deconvolution=deconv, n_base_filters=nr_base_filt,
                         n_patients_train=n_train_patients, n_patients_val=n_val_patients, patients=set_patients,
                         data_dir=data_dir, predictions_dir=predictions_dir, models_dir=models_dir, dataset=dataset,
                         result_file=result_files_dict[dataset][0],
                         result_file_per_patient=result_files_dict[dataset][1],
                         label_load_name=label_file_name, washed=False, washing_version=washing_version,
                         mode='voxel-wise')
            evaluate_set(version=version, n_epochs=selected_epoch, batch_size=bs, learning_rate=lr, dropout_rate=dr,
                         l1=l1,
                         l2=l2, batch_normalization=bn, deconvolution=deconv, n_base_filters=nr_base_filt,
                         n_patients_train=n_train_patients, n_patients_val=n_val_patients, patients=set_patients,
                         data_dir=data_dir, predictions_dir=predictions_dir, models_dir=models_dir, dataset=dataset,
                         result_file=result_files_dict_washed[dataset][0],
                         result_file_per_patient=result_files_dict_washed[dataset][1], label_load_name=label_file_name,
                         washed=True, washing_version=washing_version, mode='voxel-wise')
            evaluate_set(version=version, n_epochs=selected_epoch, batch_size=bs, learning_rate=lr,
                         dropout_rate=dr, l1=l1, l2=l2, batch_normalization=bn, deconvolution=deconv,
                         n_base_filters=nr_base_filt, n_patients_train=n_train_patients, n_patients_val=n_val_patients,
                         patients=set_patients, data_dir=data_dir, predictions_dir=predictions_dir,
                         models_dir=models_dir, dataset=dataset, result_file=result_files_dict_segment_wise[dataset][0],
                         result_file_per_patient=result_files_dict_segment_wise[dataset][1],
                         label_load_name=label_file_name, washed=True, washing_version=washing_version,
                         mode='segment-wise', segment_load_name='_skel_seg_y.nii.gz')
            print('_' * 100)
            print('Evaluation in %s dataset completed.' % dataset)
            print('_' * 100)
        K.clear_session()
    return True


def get_max_values(patients, data_dir, feature_file_names):
    print('Getting max values for normalization...')
    maxs_dict = {}
    for i, patient in enumerate(patients):
        if i % 100 == 0:
            print(str(i), '/', len(patients))
        features = [helper.load_nifti_mat_from_file(
            os.path.join(data_dir, patient + feature_file_name), printout=False) for feature_file_name
            in feature_file_names]
        num_channels = len(features)
        for ch in range(num_channels):
            if ch not in maxs_dict.keys():
                maxs_dict[ch] = []
            maxs_dict[ch].append(np.max(features[ch]))
    max_list = [max(maxs_list) for ch, maxs_list in maxs_dict.items()]
    print('Max values per channel:', max_list)
    return max_list


def load_model_and_metadata(run_name, models_dir):
    """
    Loads model and metadata in given directory with given run name.

    :param run_name: The name of the training run. String.
    :param models_dir: The path to the directory where the models are saved. String.
    :return: Tuple of trained Keras model and metadata to the model.
    """

    # Loading model.
    model_filepath = bravenet_config.get_model_filepath(models_dir=models_dir, run_name=run_name)
    print('Model file path:', model_filepath)
    model = load_model(model_filepath, compile=False)

    # Loading training metadata.
    train_metadata_filepath = bravenet_config.get_train_metadata_filepath(models_dir=models_dir, run_name=run_name)
    train_metadata = helper.load_model_metadata(train_metadata_filepath)

    return model, train_metadata


if __name__ == '__main__':
    main()
