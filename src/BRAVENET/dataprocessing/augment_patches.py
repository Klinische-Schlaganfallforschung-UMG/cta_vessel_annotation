"""
This script augments the original training patches with either rotation, or contrast or both.
Rotation can be random or by given angle and axes. When the angle is set to random, set the angle range from which the
random rotate angle is drawn.
Contrast factor can be random or specified. When the factor is set to random, set the factor range from which the
random factor is drawn.

File: augment_patches.py
Author: Jana Rieger
Created on: 21.04.2020
"""

import os

import numpy as np
from termcolor import colored

from src.BRAVENET import bravenet_config
from src.generalutils import helper

print('augment_patches.py', os.getcwd())
np.random.seed(1)

########################################
# SET
########################################
inflate_factor = 3
rotate = True
rotate_axis = 'random'  # int(1, 2 or 3) or 'random'.
rotate_angle = 'random'  # int or 'random'. in degrees.
rotate_angle_range = (-60, 60)  # set when rotate angle is random.
contrast = False
contrast_factor = 'random'  # float. 0.0 gives a solid grey matrix. 1.0 gives the original contrast.
contrast_factor_range = (0.75, 2)  # set when contrast factor is random.
########################################

# directories, paths and file names
dataset = 'train'
patient_files = bravenet_config.PATIENTS
patch_directory = bravenet_config.SAMPLES_PATH
listdir = os.listdir(patch_directory)
patch_size = (bravenet_config.PATCH_SIZE_X, bravenet_config.PATCH_SIZE_Y, bravenet_config.PATCH_SIZE_Z)
feature_files = helper.filter_list_directory(listdir, '_features', patch_size[0] * 2, patient_files[dataset]['working'],
                                             no_aug=True)
label_files = helper.filter_list_directory(listdir, '_label', patch_size[0] * 2, patient_files[dataset]['working'],
                                             no_aug=True)
n_patients = helper.number_of_patients(dataset, feature_files, label_files)

for file_idx in range(len(feature_files)):
    print('Patch', file_idx)
    if not helper.files_match(feature_files[file_idx], label_files[file_idx], filename1_split='_features_',
                              filename2_split='_label_'):
        raise AssertionError('The features file name and label file name do not match.')

    # Load features.
    features = helper.load_npz(os.path.join(patch_directory, feature_files[file_idx]))
    # Load label.
    label = helper.load_npz(os.path.join(patch_directory, label_files[file_idx]))

    for i in range(inflate_factor - 1):
        features_augmented = features.copy()
        label_augmented = label.copy()

        # Rotate patches.
        if rotate:
            features_augmented, label_augmented = helper.rotate_3d(features_augmented, label_augmented,
                                                                   rotate_axis=rotate_axis, rotate_angle=rotate_angle,
                                                                   angle_range=rotate_angle_range)

        # Change contrast in image patches.
        if contrast:
            features_augmented[..., 0] = helper.adjust_contrast(features_augmented[..., 0], factor=contrast_factor,
                                                                factor_range=contrast_factor_range, preserve_range=True)

        # Sanity checks.
        f0_unique= np.unique(features[..., 0])
        f0_aug_unique = np.unique(features_augmented[..., 0])
        if not len(f0_unique) >= len(f0_aug_unique):
            print(f0_unique, len(f0_unique))
            print(f0_aug_unique, len(f0_aug_unique))
            raise AssertionError(
                'There are more values in the augmented image than in the original image.')
        f1_unique = np.unique(features[..., 1])
        f1_aug_unique = np.unique(features_augmented[..., 1])
        if not np.array_equal(f1_unique, f1_aug_unique):
            if 0 > len(f1_unique) - len(f1_aug_unique) > 3:
                print(f1_unique, len(f1_unique))
                print(f1_aug_unique, len(f1_aug_unique))
                raise AssertionError(
                    'Values in the augmented radius are not matching values in the original radius.')
        l_unique = np.unique(label)
        l_aug_unique = np.unique(label_augmented)
        if not np.array_equal(l_unique, l_aug_unique):
            if 0 > len(l_unique) - len(l_aug_unique) > 2:
                print(l_unique, len(l_unique))
                print(l_aug_unique, len(l_aug_unique))
                raise AssertionError('Values in the augmented label are not matching values in the original label.')

        # Save augmented files.
        if rotate or contrast:
            f_path_split = feature_files[file_idx].split('.')
            l_path_split = label_files[file_idx].split('.')
            f_save_path = os.path.join(patch_directory, f_path_split[0] + '_aug' + str(i + 1))
            l_save_path = os.path.join(patch_directory, l_path_split[0] + '_aug' + str(i + 1))
            helper.save_npz(features_augmented, f_save_path, title='Augmented feature patch')
            helper.save_npz(label_augmented, l_save_path, title='Augmented label patch')
        else:
            print(colored('No augmentation!', 'red'))
    print('-' * 100)
print('DONE')
