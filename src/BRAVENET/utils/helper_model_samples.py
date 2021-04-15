"""
This script contains functions for retrieving train and val sample lists for training.

File: feature_label_lists.py
Author: Jana Rieger
Created on: 19.05.2020
"""

import os

from src.generalutils import helper

print('feature_label_lists.py', os.getcwd())


def train_feature_label_lists(listdir, patch_size, patients, augmentation):
    """
    Filters samples directory to retrieve the training samples list.

    :param listdir: List of strings. List of all samples file names in the samples directory.
    :param patch_size: Positive int. x patch size.
    :param patients: List of strings. List of training patient names.
    :param augmentation: Boolean. True for retrieve also augmented samples.
    :return: Lists of training samples file names.
    """
    train_feature_files = None
    train_label_files = None
    train_label_files_big = None

    if augmentation:
        train_feature_files_big = helper.filter_list_directory(listdir, '_features', patch_size[0] * 2,
                                                               patients)
        train_label_files_big = helper.filter_list_directory(listdir, '_label', patch_size[0] * 2, patients)
    else:
        train_feature_files = helper.filter_list_directory(listdir, '_features', patch_size[0], patients)
        train_label_files = helper.filter_list_directory(listdir, '_label', patch_size[0], patients)
        train_feature_files_big = helper.filter_list_directory(listdir, '_features', patch_size[0] * 2,
                                                               patients, no_aug=True)

    return train_feature_files, train_label_files, train_feature_files_big, train_label_files_big


def val_feature_label_lists(listdir, patch_size, patients):
    """
    Filters samples directory to retrieve the validation samples list.

    :param listdir: List of strings. List of all samples file names in the samples directory.
    :param patch_size: Positive int. x patch size.
    :param patients: List of strings. List of validation patient names.
    :return: Lists of validation samples file names.
    """
    val_feature_files_big = helper.filter_list_directory(listdir, '_features', patch_size[0] * 2, patients,
                                                             no_aug=True)
    val_label_files_big = None
    val_feature_files = helper.filter_list_directory(listdir, '_features', patch_size[0], patients)
    val_label_files = helper.filter_list_directory(listdir, '_label', patch_size[0], patients)

    return val_feature_files, val_label_files, val_feature_files_big, val_label_files_big
