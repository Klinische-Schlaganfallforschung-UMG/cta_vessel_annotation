"""
This script contains a function that trains a model with given parameters and saves it.
"""
import os
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint

from src.BRAVENET import bravenet_config
from src.BRAVENET.utils.architectures.bravenet import get_bravenet
from src.BRAVENET.utils.architectures.helper_architecture import ds_loss
from src.generalutils import helper
from src.BRAVENET.utils.datagenerator import DataGenerator

print("TensorFlow version: ", tf.__version__)


def train(version, n_epochs, batch_size, lr, dr, l1, l2, bn, deconvolution, n_base_filters,
          depth, filter_size, activation, final_activation, n_classes, optimizer, loss_function, metrics,
          n_train_patients, n_val_patients, checkpoint_model, models_dir,
          train_feature_files, train_label_files, val_feature_files, val_label_files,
          train_feature_files_big=None, train_label_files_big=None, val_feature_files_big=None,
          val_label_files_big=None, normalize_features=True,
          max_values_for_normalization=None, transfer_learning=False, transfer_weights_path=None):
    """
    Trains one model with given samples and given parameters and saves it.

    :param version: The name describes the net version. String.
    :param n_epochs: The number of epochs. Positive integer.
    :param batch_size: The size of one mini-batch. Positive integer.
    :param lr: The learning rate. Positive float.
    :param dr: The dropout rate. Positive float or None.
    :param l1: The L1 regularization. Positive float or None.
    :param l2: The L2 regularization. Positive float or None.
    :param bn: True for training with batch normalization. Boolean.
    :param deconvolution: True for using deconvolution instead of up-sampling layer. Boolean.
    :param n_base_filters: The number of filters in the first convolutional layer of the net. Positive integer.
    :param depth: The number of levels of the net. Positive integer.
    :param filter_size: The size of the 3D convolutional filters. Tuple of three positive integers.
    :param activation: The activation after the convolutional layers. String.
    :param final_activation: The activation in the final layer. String.
    :param n_classes: The number of class labels to be predicted. Positive integer.
    :param optimizer: The optimization algorithm used for training. E.g. Adam. String.
    :param loss_function: The loss function. String.
    :param metrics: List of metrics (i.e. performance measures). List of strings.
    :param n_train_patients: The number of training samples. Positive integer.
    :param n_val_patients: The number of validation samples. Positive integer.
    :param checkpoint_model: True for saving the model after each epochs during the training. Boolean.
    :param models_dir: String. Directory path where the model will be stored.
    :param train_feature_files: List of file names containing features from training set. List of strings.
    :param train_label_files: List of file names containing labels from training set. List of strings.
    :param val_feature_files: List of file names containing features from validation set. List of strings.
    :param val_label_files: List of file names containing labels from validation set. List of strings.
    :param train_feature_files_big: List of file names containing features in double-sized volume from training set.
    List of strings.
    :param train_label_files_big: List of file names containing labels in double-sized volume from training set.
    List of strings.
    :param val_feature_files_big: List of file names containing features in double-sized volume from validation set.
    List of strings.
    :param val_label_files_big: List of file names containing labels in double-sized volume from validation set.
    List of strings.
    :param normalize_features: True for scale input data between 0 an 1.
    :param max_values_for_normalization: Max values for scaling.
    :param transfer_learning: True for initialize network with pretrained weights.
    :param transfer_weights_path: Path to pretrained weights.
    """
    print('network version', version)
    print('number of epochs', n_epochs)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('dropout rate', dr)
    print('L1', l1)
    print('L2', l2)
    print('batch normalization', bn)
    print('deconvolution', deconvolution)

    # Get number of training and validation samples.
    n_train_samples = len(train_feature_files) if train_feature_files else len(train_feature_files_big)
    n_val_samples = len(val_feature_files) if val_feature_files else len(val_feature_files_big)

    # -----------------------------------------------------------
    # CREATING NAME OF CURRENT RUN
    # -----------------------------------------------------------
    run_name = bravenet_config.get_run_name(version=version, n_epochs=n_epochs, batch_size=batch_size, learning_rate=lr,
                                            dropout_rate=dr, l1=l1, l2=l2, batch_normalization=bn,
                                            deconvolution=deconvolution, n_base_filters=n_base_filters,
                                            n_patients_train=n_train_patients, n_patients_val=n_val_patients)
    formatting_run_name = bravenet_config.get_run_name(version=version, n_epochs=n_epochs, batch_size=batch_size,
                                                       learning_rate=lr, dropout_rate=dr, l1=l1, l2=l2,
                                                       batch_normalization=bn, deconvolution=deconvolution,
                                                       n_base_filters=n_base_filters, n_patients_train=n_train_patients,
                                                       n_patients_val=n_val_patients, formatting_epoch=True)
    # File paths.
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_filepath = bravenet_config.get_model_filepath(models_dir, run_name)
    train_metadata_filepath = bravenet_config.get_train_metadata_filepath(models_dir, run_name)
    train_history_filepath = bravenet_config.get_train_history_filepath(models_dir, run_name)
    logdir = bravenet_config.LOG_DIR

    # -----------------------------------------------------------
    # CREATING MODEL
    # -----------------------------------------------------------
    input_shape = (bravenet_config.PATCH_SIZE_X, bravenet_config.PATCH_SIZE_Y, bravenet_config.PATCH_SIZE_Z,
                   bravenet_config.NUM_FEATURES)
    # Double all dimensions except the last one because that is the number of feature channels.
    input_shape_big = tuple(v * 2 if i < len(input_shape) - 1 else v for i, v in enumerate(input_shape))
    num_outputs = depth - 1

    # Load specific architectures according to the model version.
    model = get_bravenet(input_shapes=[input_shape_big, input_shape], n_classes=n_classes,
                         activation=activation, final_activation=final_activation, n_base_filters=n_base_filters,
                         depth=depth, optimizer=optimizer, learning_rate=lr, dropout=dr, l1=l1, l2=l2,
                         batch_normalization=bn, loss_function=loss_function, metrics=metrics, filter_size=filter_size,
                         deconvolution=deconvolution)

    # -----------------------------------------------------------
    # TRAINING MODEL
    # -----------------------------------------------------------
    starttime_training = helper.start_time_measuring('training')

    # SET CALLBACKS
    callbacks = []
    # keras callback for tensorboard logging
    tb = TensorBoard(log_dir=logdir, histogram_freq=1)
    callbacks.append(tb)
    # keras callback for saving the training history to csv file
    csv_logger = CSVLogger(train_history_filepath)
    callbacks.append(csv_logger)
    # keras ModelCheckpoint callback saves the model after every epoch, monitors the val_dice and does not overwrite
    # if the val_dice gets worse
    if checkpoint_model:
        mc = ModelCheckpoint(bravenet_config.get_model_filepath(models_dir, formatting_run_name),
                             monitor='val_loss', verbose=1, save_best_only=False,
                             mode='min')
        callbacks.append(mc)

    # DATAGENERATORS
    train_generator = DataGenerator(train_feature_files, train_feature_files_big, train_label_files,
                                    train_label_files_big, bravenet_config.SAMPLES_PATH, batch_size=batch_size,
                                    dim=input_shape[:-1], dim_big=input_shape_big[:-1],
                                    n_channels=bravenet_config.NUM_FEATURES, num_outputs=num_outputs, shuffle=True,
                                    normalize_features=normalize_features,
                                    max_values_for_normalization=max_values_for_normalization)
    val_generator = DataGenerator(val_feature_files, val_feature_files_big, val_label_files, val_label_files_big,
                                  bravenet_config.SAMPLES_PATH, batch_size=batch_size, dim=input_shape[:-1],
                                  dim_big=input_shape_big[:-1], n_channels=bravenet_config.NUM_FEATURES,
                                  num_outputs=num_outputs, shuffle=True, normalize_features=normalize_features,
                                  max_values_for_normalization=max_values_for_normalization)

    # TRANSFER LEARNING
    if transfer_learning:
        # Load weights
        # untrained_model = clone_model(model)
        model.load_weights(transfer_weights_path, by_name=True)
        print('Weights loaded.')
        # Multiple loss
        loss, loss_weights = ds_loss(depth=depth, loss_function=loss_function)
        model.compile(optimizer=optimizer(lr=lr), loss=loss, metrics=metrics, loss_weights=loss_weights)
        # untrained_model.compile(optimizer=optimizer(lr=lr), loss=loss, metrics=metrics, loss_weights=loss_weights)
        print('model compiled.')

    # TRAIN
    history = None
    try:
        history = model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            steps_per_epoch=n_train_samples // batch_size,
            validation_steps=n_val_samples // batch_size,
            epochs=n_epochs,
            verbose=2, shuffle=True, callbacks=callbacks)
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        exit(0)
    finally:
        if history is not None:
            duration_training = helper.end_time_measuring(starttime_training, 'training')

            # SAVING MODEL AND PARAMS
            if checkpoint_model:
                print('Model was checkpointed -> not saving the model from last epoch.')
            else:
                print('Model was not checkpointed -> saving the model from last epoch to:', model_filepath)
                model.save(model_filepath)

            print('Saving params to ', train_metadata_filepath)
            history.params['version'] = version
            history.params['batchsize'] = batch_size
            history.params['learning_rate'] = lr
            history.params['dropout_rate'] = dr
            history.params['l1'] = l1
            history.params['l2'] = l2
            history.params['batch_norm'] = bn
            history.params['deconvolution'] = deconvolution
            history.params['num_base_filters'] = n_base_filters
            history.params['loss'] = loss_function
            history.params['samples'] = n_train_samples
            history.params['val_samples'] = n_val_samples
            history.params['total_time'] = duration_training
            history.params['normalize features'] = normalize_features
            history.params['max_values_for_normalization'] = max_values_for_normalization
            results = {'params': history.params, 'history': history.history}
            with open(train_metadata_filepath, 'wb') as handle:
                pickle.dump(results, handle)
    return history
