"""
This script extracts 3D patches randomly equally around all class labels (except background and not-annotated class
label). There are 2 features (MRA image, skeleton with radius values) and corresponding label (annotated skeleton)
patches extracted. At each location there is one patch in given arbitrarily size extracted and one patches in
double-size in all dimensions. Both patches are centered around the randomly chosen voxel.
"""

import os
import time

import numpy as np

from src.BRAVENET import bravenet_config
from src.generalutils import helper

print('3d_patch_extraction_double_size', os.getcwd())

########################################
# SET
########################################
datasets = ['train', 'val', 'test']
feature_dtype = 'float32'
label_dtype = 'uint8'
csv_file = 'src/BRAVENET/analyses/number_of_class_voxels.csv'
nr_of_patches_around_each_class = 8  # number of patches we want to extract from each patient around each class
size_multipliers = [1, 2]
########################################

class_dict = bravenet_config.CLASS_DICTS[bravenet_config.H_LEVEL]
# directories, paths and file names
feature_file_names = bravenet_config.FEATURE_FILENAMES
label_name = bravenet_config.LABEL_FILENAMES[bravenet_config.H_LEVEL]
data_dir = bravenet_config.DATA_DIR
patient_files = bravenet_config.PATIENTS

patch_size_x = bravenet_config.PATCH_SIZE_X
patch_size_y = bravenet_config.PATCH_SIZE_Y
patch_size_z = bravenet_config.PATCH_SIZE_Z
num_features = bravenet_config.NUM_FEATURES

half_patch_size_x = patch_size_x // 2
half_patch_size_y = patch_size_y // 2
half_patch_size_z = patch_size_z // 2

# classes without background and not-annotated class
classes = [*class_dict][2:]
helper.write_to_csv(csv_file, [['patient'] + classes])

print('Number of patches to extract per class per patient:', nr_of_patches_around_each_class)

# extract patches from each data stack (patient)
starttime_total = helper.start_time_measuring(what_to_measure='total extraction')
for dataset in datasets:
    patients = patient_files[dataset]
    # check number of patients in datasets
    helper.check_num_patients(patients)
    all_patients = patients['working']
    num_patients = len(all_patients)
    print('Number of patients:', num_patients)
    patch_directory = bravenet_config.SAMPLES_PATH
    if not os.path.exists(patch_directory):
        os.makedirs(patch_directory)

    for patient in all_patients:
        print('DATA SET:', dataset)
        print('PATIENT:', patient)
        patient_class_list = []

        # load data
        print('> Loading features...')
        features = [helper.load_nifti_mat_from_file(
            os.path.join(data_dir, patient + feature_file_name)) for feature_file_name in feature_file_names]
        print('> Loading label...')
        label = helper.load_nifti_mat_from_file(os.path.join(data_dir, patient + label_name))
        print('Patient ', patient, ': features and label loaded.')
        helper.check_dimensions(features, label.shape)

        # sizes
        min_x = 0
        min_y = 0
        min_z = 0
        max_x = label.shape[0]
        max_y = label.shape[1]
        max_z = label.shape[2]

        # for each class except background and not-annotated extract given number of patches
        patch = 0
        for cls in classes:
            print('CLASS', cls)
            duration = 0

            # find indices of all voxels belonging to current class
            class_inds = np.asarray(np.where(label == cls))
            print('class inds', class_inds)
            nr_of_class_voxels = len(class_inds[0])
            patient_class_list.append(nr_of_class_voxels)
            # some classes might be missing in some patients therefore check if the current class is present
            if nr_of_class_voxels > 0:
                # if there are not enough class voxels
                if nr_of_class_voxels <= nr_of_patches_around_each_class:
                    nr_of_samples = nr_of_class_voxels
                    samples_inds = class_inds
                else:
                    nr_of_samples = nr_of_patches_around_each_class
                    # choose given number of random class indices
                    random_inds_of_class_inds = np.random.choice(class_inds.shape[1], nr_of_patches_around_each_class,
                                                                 replace=False)
                    random_class_inds = class_inds[:, random_inds_of_class_inds]
                    samples_inds = random_class_inds
                # extract patches
                for voxel in range(nr_of_samples):
                    x = samples_inds[0, voxel]
                    y = samples_inds[1, voxel]
                    z = samples_inds[2, voxel]
                    for size_multiplier in size_multipliers:
                        feature_patch = np.zeros((patch_size_x * size_multiplier,
                                                  patch_size_y * size_multiplier,
                                                  patch_size_z * size_multiplier,
                                                  bravenet_config.NUM_FEATURES), dtype=feature_dtype)
                        label_patch = np.zeros((patch_size_x * size_multiplier,
                                                patch_size_y * size_multiplier,
                                                patch_size_z * size_multiplier), dtype=label_dtype)
                        # find the starting and ending x and y coordinates of given patch
                        patch_start_x = max(x - half_patch_size_x * size_multiplier, min_x)
                        patch_end_x = min(x + half_patch_size_x * size_multiplier, max_x)
                        patch_start_y = max(y - half_patch_size_y * size_multiplier, min_y)
                        patch_end_y = min(y + half_patch_size_y * size_multiplier, max_y)
                        patch_start_z = max(z - half_patch_size_z * size_multiplier, min_z)
                        patch_end_z = min(z + half_patch_size_z * size_multiplier, max_z)
                        # if the class voxel is near the volume starting edge, add zero padding
                        offset_x = max(min_x - (x - half_patch_size_x * size_multiplier), 0)
                        offset_y = max(min_y - (y - half_patch_size_y * size_multiplier), 0)
                        offset_z = max(min_z - (z - half_patch_size_z * size_multiplier), 0)
                        # extract patches from all features and label
                        for f in range(num_features):
                            feature_patch[offset_x: offset_x + (patch_end_x - patch_start_x),
                            offset_y: offset_y + (patch_end_y - patch_start_y),
                            offset_z: offset_z + (patch_end_z - patch_start_z), f] \
                                = features[f][patch_start_x:patch_end_x, patch_start_y:patch_end_y,
                                  patch_start_z:patch_end_z]
                        label_patch[offset_x: offset_x + (patch_end_x - patch_start_x),
                        offset_y: offset_y + (patch_end_y - patch_start_y),
                        offset_z: offset_z + (patch_end_z - patch_start_z)] \
                            = label[patch_start_x:patch_end_x, patch_start_y:patch_end_y, patch_start_z:patch_end_z]
                        # save patches
                        start_time = time.time()
                        f_save_path = os.path.join(patch_directory,
                                                   patient + '_class_' + str(cls) + '_features_' + str(
                                                       patch) + '_size_' + str(patch_size_x * size_multiplier))
                        helper.save_npz(np.asarray(feature_patch), f_save_path, title='Feature patch')
                        l_save_path = os.path.join(patch_directory,
                                                   patient + '_class_' + str(cls) + '_label_' + str(
                                                       patch) + '_size_' + str(patch_size_x * size_multiplier))
                        helper.save_npz(np.asarray(label_patch), l_save_path, title='Label patch')
                        end_time = time.time()
                        duration += end_time - start_time
                    # next patch
                    patch += 1
            # print('duration gzip', duration_gzip)
            print('duration', duration)
        helper.write_to_csv(csv_file, [[patient] + patient_class_list])
        print('---------------------------------------------------------------------')
helper.end_time_measuring(starttime_total, what_to_measure='total extraction')
print('DONE')
