"""
This file contains helper functions for other scripts.
"""

import csv
import gzip
import json
import os
import pickle
import sys
import time
from functools import partial, update_wrapper
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import rotate


def load_nifti_mat_from_file(path_orig: str, dtype=None, printout=True):
    """
    Loads a nifti file and returns the data from the nifti file as numpy array.
    :param path_orig: Path from where to load the nifti. String.
    :param dtype: Data type of the output array. String.
    :param printout: True for print out info to console. Boolean.
    :return: Nifti data. Numpy array.
    """
    nifti_orig = nib.load(path_orig)
    if printout:
        print('Nifti loaded from:', path_orig)
        print(' - dimensions: ', nifti_orig.shape)
        print(' - original data type:', nifti_orig.get_data_dtype())
    matrix = nifti_orig.get_data()
    if dtype is not None:
        matrix = np.array(matrix, dtype=dtype)  # transform the images into np.ndarrays with given dtype
        if printout:
            print(' - new data type:', matrix.dtype)
    return matrix


def create_and_save_nifti(mat, path_target: str):
    """
    Creates a nifti image from numpy array and saves it to given path.
    :param mat: Matrix to save. Numpy array.
    :param path_target: Path where to store the created nifti. String.
    """
    new_nifti = nib.Nifti1Image(mat, np.eye(4))  # create new nifti from matrix
    nib.save(new_nifti, path_target)  # save nifti to target dir
    print('New nifti saved to:', path_target)
    print(' - dimensions: ', new_nifti.shape)
    print(' - data type:', new_nifti.get_data_dtype())


def load_json(path_orig: str, printout=True):
    with open(path_orig, 'r') as f:
        json_data = json.load(f)
        if printout:
            print('Json loaded from:', path_orig)
    return json_data


def save_json(data, path_target: str):
    """
    Saves json data to file at given path.
    :param data: Json to save.
    :param path_target: Path where to store the json data. String.
    """
    with open(path_target, 'w') as outfile:
        json.dump(data, outfile)
    print('Json saved to:', path_target)
    return True


def load_feature_volumes(feature_file_names, data_dir, patient, num_features_for_check):
    """
    Loads given list of feature files and returns a list of feature volumes.

    :param feature_file_names: List of feature file names to load. List of strings.
    :param data_dir: The path to data directory where the features are saved. String.
    :param patient: The name of the patient for which to load the features. String.
    :param num_features_for_check: Number of features to load for sanity check. Positive integer.
    :return: List of feature volumes. List of 3D ndarrays.
    """
    print('patient:', patient)
    print('feature file names', feature_file_names)

    if len(feature_file_names) != num_features_for_check:
        raise AssertionError('The number of features is not equal to NUM_FEATURES in *config.py file.')

    print('> Loading features...')
    feature_volumes = [load_nifti_mat_from_file(os.path.join(data_dir, patient + f)) for f in
                       feature_file_names]

    return feature_volumes


def load_model_metadata(train_metadata_filepath):
    try:
        with open(train_metadata_filepath, 'rb') as handle:
            train_metadata = pickle.load(handle)
        print('Train params:', train_metadata['params'])
    except FileNotFoundError:
        train_metadata = None
        print('Unexpected error by reading train metadata:', sys.exc_info()[0])
    return train_metadata


def check_num_patients(patients):
    # the number of patient must be same in every category (original, working, working augmented) defined in the patient
    # dictionary in the config.py
    num_patients = len(patients['original'])
    if num_patients != len(patients['working']):
        raise AssertionError('Check config.PATIENT_FILES. The lists with patient names do not have the same length.')
    return True


def check_dimensions(mat_list, dimensions):
    # Checks if all of the matrices in the list have same dimensions.
    same_dimensions = True
    for i in range(len(mat_list) - 1):
        if mat_list[i].shape != mat_list[i + 1].shape:
            same_dimensions = False
    if not same_dimensions:
        raise AssertionError('The DIMENSIONS of the volumes are NOT SAME.')

    # Checks if all of the matrices in the list have correct given dimensions.
    correct_dimensions = True
    wrong_shape_list = []
    wrong_mat_list = []
    for i in range(len(mat_list)):
        for j in range(len(dimensions)):
            if mat_list[i].shape[j] != dimensions[j]:
                correct_dimensions = False
                if i not in wrong_mat_list:
                    wrong_mat_list.append(i)
                    wrong_shape_list.append(mat_list[i].shape)
    for i in wrong_mat_list:
        if not correct_dimensions:
            raise AssertionError('The DIMENSIONS of the matrix ' + str(i) + ' are NOT ' + str(
                dimensions) + ' but ' + str(wrong_shape_list[i]) + '.')
    print('!!! All volumes have correct dimensions.')
    return True


def stringify(lst):
    """
    Joins list elements to one string.

    :param lst: list. List of strings.
    :return: string. Joined list of strings to one string.
    """
    return ''.join(str(x) for x in lst)


def unique_and_count(array, name='', print_dict=True):
    unique, count = np.unique(array, return_counts=True)
    print(name, 'unique:', unique, 'size:', len(unique))
    if print_dict:
        print(name, dict(zip(unique, count)))
    return unique, count


def load_npy_gzipped(path, printout='True'):
    unzipped_file = gzip.GzipFile(path, "r")
    matrix = np.load(unzipped_file)
    if printout:
        print(' - matrix loaded from:', path)
        print(' - dimensions: ', matrix.shape)
        print(' - data type:', matrix.dtype)
    return matrix


def save_npy_gzipped(array, path, title='Data'):
    f = gzip.GzipFile(path, 'w')
    np.save(f, array)
    f.close()
    print(title, 'saved to', path)
    print(' - array data type:', array.dtype)
    return True


def load_npz(path, printout='True'):
    """
    Loads ndarray from compressed file.

    :param path: string. File path.
    :param printout: boolean. Whether to print out info.
    :return: ndarray. Uncompressed matrix.
    """
    matrix = np.load(path)['arr_0']
    if printout:
        print('Data loaded from:', path)
        print(' - dimensions: ', matrix.shape)
        print(' - data type:', matrix.dtype)
    return matrix


def save_npz(array, path, title='Data'):
    """
    Saves compressed ndarray.

    :param array: ndarray. Array to compress.
    :param path: string. File path.
    :param title: string. Name of the array just for print out.
    :return: boolean. True for successful saving.
    """
    np.savez_compressed(path, np.asarray(array))
    print(title, 'saved to', path)
    print(' - array data type:', array.dtype)
    return True


def number_of_patients(dataset, feature_files, label_files):
    """
    Calculates number of unique patients in the list of given filenames.

    :param dataset: string. Dataset train/val/test.
    :param feature_files: list of strings. List of filenames with patient names containing features.
    :param label_files: list of strings. List of filenames with patient names containing labels.
    :return: int. Number of unique patients.
    """
    if len(feature_files) != len(label_files):
        raise AssertionError(dataset, 'files have different length.')
    print('Number of', dataset, 'files:', len(feature_files))

    patients = []
    for file in feature_files:
        patient = file[:4]
        if patient not in patients:
            patients.append(patient)
    print(dataset, 'patients:', patients)
    num_patients = len(patients)
    print('Num', dataset, 'patients:', num_patients)
    return num_patients


def convert_to_binary(skeleton):
    """
    Converts a matrix to binary matrix. Values greater than 0 become 1. Values lower than or equal to 0 become 0.

    :param skeleton: ndarray. Matrix to convert to binary.
    :return: ndarray. Binarized matrix.
    """
    binary_skeleton = np.zeros_like(skeleton, dtype='uint8')
    binary_skeleton[skeleton > 0] = 1
    binary_skeleton[skeleton <= 0] = 0
    return binary_skeleton


def nonzero_indices_match(list_of_skeletons):
    """
    Checks if all skeleton coordinates in the skeletons from the given list match.

    :param list_of_skeletons: list. List containing
    :return: boolean. True for skeleton coordinates match.
    """
    binary_skeletons = [convert_to_binary(skel) for skel in list_of_skeletons]
    for i in range(len(list_of_skeletons) - 1):
        if not (binary_skeletons[i] == binary_skeletons[i + 1]).all():
            raise AssertionError('The volume non-zero indices are not equal.')
    print('!!! Volume non-zero indices match.')
    return True


def start_time_measuring(what_to_measure=''):
    """
    Returns the start time of an operation.

    :param what_to_measure: string. What to measure. For print out.
    :return: localtime. Start time.
    """
    starttime = time.localtime()
    print('Start time', what_to_measure, ':', time.strftime('%a, %d %b %Y %H:%M:%S', starttime))
    return starttime


def end_time_measuring(start_time, what_to_measure='', print_end_time=True):
    """
    Returns the end time and duration of an operation.

    :param start_time: localtime. Start time of an operation.
    :param what_to_measure: string. What to measure. For print out.
    :param print_end_time: boolean. True for printing out the end time.
    :return: localtime, second. End time, duration.
    """
    endtime = time.localtime()
    if print_end_time:
        print('End time', what_to_measure, ':', time.strftime('%a, %d %b %Y %H:%M:%S', endtime))
    duration = int(time.mktime(endtime) - time.mktime(start_time))
    print(what_to_measure.capitalize(), 'took:', (duration // 3600), 'hours', (duration % 3600) // 60,
          'minutes', (duration % 3600) % 60, 'seconds')
    return endtime, duration


def write_to_csv(file, rows: list):
    """
    Write a row to given CSV file.

    :param file: string. CSV file to write in.
    :param rows: list. List of rows. Row is a list of values to write in the csv file as a row.
    :return: boolean. True for successful write.
    """
    print('Writing to csv...')
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    return True


def wash_with_segments_max_scores(softmax_scores, segment_matrix, printout=False):
    """
    Washes annotation matrix with vessels segments according to maximal softmax score in each segment.

    :param softmax_scores: Matrix with predicted softmax scores. ndarray.
    :param segment_matrix: Matrix with vessel segments. ndarray.
    :return: Washed matrix. ndarray.
    """
    print('Washing with segments...')
    washed_matrix = np.zeros_like(segment_matrix)
    unique_segments = np.unique(segment_matrix)
    for segment in unique_segments:
        # find all segment indices
        segment_inds = np.where(segment_matrix == segment)
        # sum scores of all voxels in segment per class
        segment_scores = softmax_scores[segment_inds]
        sum_scores = np.sum(segment_scores, axis=0)
        class_with_max_score = np.argmax(sum_scores)
        washed_matrix[segment_inds] = class_with_max_score
        if printout:
            print('Segment:', segment)
            print('shape segment scores:', segment_scores.shape)
            print('sum scores:', sum_scores)
            print('max score class:', class_with_max_score)
    return washed_matrix


def skeleton_radius_from_json_graph(graph_json, volume_dimensions, background_value, radius_dtype):
    """
    Converts a JSON structure with a skeleton graph to a ndarray with skeleton containing radius values.

    :param graph_json: JSON structure with vessel skeleton graph.
    :param volume_dimensions: Dimensions of the nifti volumes. Tuple of 3 integers.
    :param background_value: Number for coding the background voxels. Integer.
    :param radius_dtype: Data type of the ndarray with skeleton with radius values.
    :return: Skeleton with radius values as ndarray.
    """

    # matrix filled with background values
    skeleton_radius = np.full(shape=volume_dimensions, fill_value=background_value,
                              dtype=radius_dtype)

    # for all points in graph.json read the radius and write it to the skeleton radius matrix
    for seg in graph_json['graph']['segments']:
        for point in seg['segment']['points']:
            skeleton_radius[point['x'], point['y'], point['z']] = point['radiant']

    print('skeleton shape:', skeleton_radius.shape)
    print('skeleton dtype:', skeleton_radius.dtype)
    unique_and_count(skeleton_radius, 'skeleton radius')

    return skeleton_radius


def skeleton_segments_from_json_graph(graph_json, volume_dimensions, background_value, segments_dtype, axis):
    """
    Converts a JSON structure with a skeleton graph to a ndarray with skeleton containing vessel segment encoded with
    depth in given axis.

    :param graph_json: JSON structure with vessel skeleton graph.
    :param volume_dimensions: Dimensions of the nifti volumes. Tuple of 3 integers.
    :param background_value: Number for coding the background voxels. Integer.
    :param segments_dtype: Data type of the ndarray with skeleton with radius values.
    :param axis: Volume axis in which depth to encode the segments with unique ids.
    :return: Tuple of Skeleton with vessel segments as ndarray, Number of segments in the volume,
    Number of collisions encountered during the volume creation.
    """

    # matrix filled with background values
    skeleton_segments = np.full(shape=volume_dimensions, fill_value=background_value,
                                dtype=segments_dtype)

    # load the segments for skeleton voxels from graph.json, wash it with given axis depth and save into the
    # background matrix
    count_segments = 0
    count_collisions = 0
    min_axis_history = []
    for seg in graph_json['graph']['segments']:
        # find all axis inds in one segment
        axis_inds = []
        for point in seg['segment']['points']:
            axis_inds.append(point[axis])
        # find min axis value among the values in one segment
        min_axis_ind = min(axis_inds)
        if min_axis_ind == 0:
            min_axis_ind = 1
        min_axis_ind = min_axis_ind * 100
        # another segment was already encoded with the same number (that means another segment also started
        # in the min y slice) increase the value of min_y so long until no segment was previously
        # encoded with the same number
        first_collision_in_segment = True
        while min_axis_ind in min_axis_history:
            min_axis_ind += 1
            if first_collision_in_segment:
                count_collisions += 1
            first_collision_in_segment = False
        min_axis_history.append(min_axis_ind)
        # assign the value of min_y to each point in the segment
        for point in seg['segment']['points']:
            skeleton_segments[point['x'], point['y'], point['z']] = min_axis_ind
        count_segments += 1

    print('skeleton shape:', skeleton_segments.shape)
    print('skeleton dtype:', skeleton_segments.dtype)
    print('# of segments counted', count_segments)
    print('# of collisions counted', count_collisions)

    return skeleton_segments, count_segments, count_collisions


def patient_dict(file, sheetname, database=None):
    df = pd.read_excel(file, sheet_name=sheetname, converters={'Working ID': str})
    if database is None:
        pat_dict = {
            'train': {
                'original': df[(df['Set'] == 'train') & (df['Include'])]['Original patient ID'].tolist(),
                'working': df[(df['Set'] == 'train') & (df['Include'])]['Working ID'].tolist()
            },
            'val': {
                'original': df[(df['Set'] == 'val') & (df['Include'])]['Original patient ID'].tolist(),
                'working': df[(df['Set'] == 'val') & (df['Include'])]['Working ID'].tolist(),
            },
            'test': {
                'original': df[(df['Set'] == 'test') & (df['Include'])][
                    'Original patient ID'].tolist(),
                'working': df[(df['Set'] == 'test') & (df['Include'])][
                    'Working ID'].tolist(),
            }
        }
    else:
        pat_dict = {
            'train': {
                'original': df[(df['Database'] == database) & (df['Set'] == 'train') & (df['Include'])][
                    'Original patient ID'].tolist(),
                'working': df[(df['Database'] == database) & (df['Set'] == 'train') & (df['Include'])][
                    'Working ID'].tolist()
            },
            'val': {
                'original': df[(df['Database'] == database) & (df['Set'] == 'val') & (df['Include'])][
                    'Original patient ID'].tolist(),
                'working': df[(df['Database'] == database) & (df['Set'] == 'val') & (df['Include'])][
                    'Working ID'].tolist(),
            },
            'test': {
                'original': df[(df['Database'] == database) & (df['Set'] == 'test') & (df['Include'])][
                    'Original patient ID'].tolist(),
                'working': df[(df['Database'] == database) & (df['Set'] == 'test') & (df['Include'])][
                    'Working ID'].tolist(),
            }
        }

    return pat_dict


def half_center_matrix(matrix, dim):
    """
    Cuts out matrix of half size of first 3 dimensions from the center of the original matrix.

    :param matrix: ndarray.
    :param dim: tuple of 3 ints.
    :return: ndarray. Half-sized matrix.
    """
    central_voxel = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    quarter_size = (dim[0] // 4, dim[1] // 4, dim[2] // 4)
    down = (
        central_voxel[0] - quarter_size[0], central_voxel[1] - quarter_size[1], central_voxel[2] - quarter_size[2])
    up = (
        central_voxel[0] + quarter_size[0], central_voxel[1] + quarter_size[1], central_voxel[2] + quarter_size[2])
    return matrix[down[0]:up[0], down[1]:up[1], down[2]:up[2]]


def rotate_3d(matrix1, matrix2=None, rotate_axis='random', rotate_angle='random', angle_range=None):
    """
    Rotates a 3d matrix randomly or by given angle and axes.

    :param: rotate_axis: int(1, 2 or 3) or 'random'. 1, 2 and 3 correspond to x axis, y axis and z axis respectively.
                Axis along which the matrix is rotated.
    :param: rotate_angle: int or 'random'. Angle in degrees by which matrix is rotated along the specified axis.
    :param: angle_range: tuple of two ints. Range from which to draw the random rotate angle.
    :return: Rotated matrix.
    """
    if rotate_axis == 'random':
        rotate_axis = np.random.randint(0, 3)
    elif rotate_axis not in (0, 1, 2):
        raise ValueError('Rotate axis must be 0, 1, 2 or "random".')

    if rotate_axis == 0:
        axes_plane = (2, 1)
    elif rotate_axis == 1:
        axes_plane = (0, 2)
    elif rotate_axis == 2:
        axes_plane = (0, 1)

    if rotate_angle == 'random':
        if angle_range is None:
            rotate_angle = np.random.randint(-180, 180)
        else:
            if type(angle_range) == tuple and len(angle_range) == 2:
                rotate_angle = np.random.randint(angle_range[0], angle_range[1])
            else:
                raise ValueError('Angle range must be a tuple of two integers.')
    elif type(rotate_angle) != int:
        raise ValueError('Rotate angle must be int or "random".')
    print('Rotate angle:', rotate_angle)
    print('Rotate axis:', rotate_axis)

    rot_matrix1 = rotate(matrix1, angle=rotate_angle, axes=axes_plane, reshape=False, mode='constant', cval=0.0,
                         order=0)
    if matrix2 is not None:
        rot_matrix2 = rotate(matrix2, angle=rotate_angle, axes=axes_plane, reshape=False, mode='constant', cval=0.0,
                             order=0)
        return rot_matrix1, rot_matrix2
    return rot_matrix1


def adjust_contrast(matrix, factor='random', factor_range=(0.75, 5), preserve_range=False):
    """
    Adjusts contrast in matrix.

    :param matrix: ndarray. The matrix where to change the contrast.
    :param factor: float. An enhancement factor of 0.0 gives a solid grey matrix. A factor of 1.0 gives the original
    contrast.
    :param factor_range: tuple of floats. Range from where to draw a random enhancement factor when factor parameter
    is set to 'random'.
    :param preserve_range: boolean. True for preserving the same value ranges in the matrix.
    :return: ndarray. Contrast adjusted matrix.
    """
    if factor == 'random':
        if type(factor_range) == tuple and len(factor_range) == 2:
            factor = np.random.uniform(factor_range[0], factor_range[1])
        else:
            raise ValueError('Contrast range must be a tuple of two floats.')
    else:
        factor = float(factor)
    print('Contrast factor:', factor)
    matrix_mean = matrix.mean()
    contrast_matrix = (matrix - matrix_mean) * factor + matrix_mean

    if preserve_range:
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        contrast_matrix[contrast_matrix < matrix_min] = matrix_min
        contrast_matrix[contrast_matrix > matrix_max] = matrix_max

    return contrast_matrix


def files_match(filename1, filename2, filename1_split='_features_', filename2_split='_label_'):
    f1_split = filename1.split(filename1_split)
    f2_split = filename2.split(filename2_split)
    match = f1_split == f2_split
    if not match:
        print(filename1)
        print(filename2)
    return match


def filter_list_directory(listdir, filetype, patch_size, patient_list, no_aug=False):
    """
    Filters filenames in a directory.

    :param listdir: list. Directory content.
    :param filetype: string. '_features' / '_labels'.
    :param patch_size: int. Patch size.
    :param patient_list: list. List of patient names.
    :param no_aug: boolean. True for only not augmented files.
    :return: list. Filtered list of filenames.
    """
    if no_aug:
        filtered_list = [file for file in listdir if
                         filetype in file and
                         '_size_' + str(patch_size) in file and
                         file[:4] in patient_list and
                         '_aug' not in file]
    else:
        filtered_list = [file for file in listdir if
                         filetype in file and
                         '_size_' + str(patch_size) in file and
                         file[:4] in patient_list]
    filtered_list.sort()
    return filtered_list


# testing and debugging
if __name__ == '__main__':

    for i in [0, 2, 4, 6]:
        print('division', int((i + 2) / 2))
    for i in [1, 3, 5, 7]:
        print('division', int((i + 1) / 2))

    softmax_scores = np.array([[[[0.1, 0.5, 0.3, 0.1]], [[0.2, 0.1, 0.6, 0.1]]],
                               [[[0.4, 0.4, 0.1, 0.1]], [[0.2, 0.2, 0.3, 0.3]]],
                               [[[0.1, 0.5, 0.2, 0.2]], [[0.2, 0.3, 0.1, 0.4]]]])
    print('--', softmax_scores.shape)
    segment_matrix = np.array([[[1], [1]],
                               [[2], [3]],
                               [[4], [3]]])
    print('--', segment_matrix.shape)
    wash_with_segments_max_scores(softmax_scores, segment_matrix)

