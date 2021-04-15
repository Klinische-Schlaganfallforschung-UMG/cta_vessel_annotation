"""
This script contains the function that divides the input volumes in small patches, predicts the patches with given
model and saves the predicted annotated volume as nifti file.
"""

import sys

import numpy as np
from tensorflow.keras.models import load_model

from src.BRAVENET import bravenet_config
from src.generalutils import helper


def predict(feature_volumes, model, num_classes, patch_size_x, patch_size_y, patch_size_z, annotation_save_path,
            max_values_for_normalization=None, skeleton_segments=None, washed_annotation_save_path=None):
    """
    :param feature_volumes: List of input feature volumes. List of 3D ndarrays.
    :param model: Trained Keras model.
    :param num_classes: Number of classes to predict. Positive integer.
    :param patch_size_x: Size of the patch in x axis. Positive integer.
    :param patch_size_y: Size of the patch in y axis. Positive integer.
    :param patch_size_z: Size of the patch in z axis. Positive integer.
    :param annotation_save_path: Path (including the file name) where to save the annotation results as nifti. String.
    :param max_values_for_normalization: Max values for scaling between 0 and 1.
    :param skeleton_segments: Volume with vessel segments.
    :param washed_annotation_save_path: Path to predicted washed volume.
    :return: Predicted annotated volume. 3D ndarray.
    """
    normalize = max_values_for_normalization is not None

    num_features = len(feature_volumes)
    volume_dimensions = feature_volumes[0].shape
    min_x = 0
    min_y = 0
    min_z = 0
    max_x = volume_dimensions[0]
    max_y = volume_dimensions[1]
    max_z = volume_dimensions[2]
    num_x_patches = int(np.ceil((max_x - min_x) / float(patch_size_x)))
    num_y_patches = int(np.ceil((max_y - min_y) / float(patch_size_y)))
    num_z_patches = int(np.ceil((max_z - min_z) / float(patch_size_z)))

    print('volume dimensions', volume_dimensions)
    print('num x patches', num_x_patches)
    print('num y patches', num_y_patches)
    print('num z patches', num_z_patches)

    # the predicted annotation is going to be saved in this probability matrix
    predicted_probabilities = np.zeros(volume_dimensions + (num_classes,), dtype='float32')

    # start cutting out and predicting the patches
    starttime_volume = helper.start_time_measuring(what_to_measure='volume prediction')
    p = 0
    for ix in range(num_x_patches):
        for iy in range(num_y_patches):
            for iz in range(num_z_patches):
                # find the starting and ending coordinates of the given patch
                patch_start_x = patch_size_x * ix
                patch_end_x = min(patch_size_x * (ix + 1), max_x)
                patch_start_y = patch_size_y * iy
                patch_end_y = min(patch_size_y * (iy + 1), max_y)
                patch_start_z = patch_size_z * iz
                patch_end_z = min(patch_size_z * (iz + 1), max_z)
                # extract patch for prediction
                patch = np.zeros((1, patch_size_x, patch_size_y, patch_size_z, num_features), dtype='float32')
                for f in range(num_features):
                    patch[0, :patch_end_x - patch_start_x, :patch_end_y - patch_start_y, :patch_end_z - patch_start_z,
                          f] = feature_volumes[f][patch_start_x:patch_end_x, patch_start_y:patch_end_y,
                                                  patch_start_z:patch_end_z].astype('float32')
                # normalize patch
                if normalize:
                    assert isinstance(max_values_for_normalization, list)
                    for f in range(num_features):
                        patch[..., f] = patch[..., f] / max_values_for_normalization[f]

                # find center location in the patch
                center_x = patch_start_x + patch_size_x // 2
                center_y = patch_start_y + patch_size_y // 2
                center_z = patch_start_z + patch_size_z // 2
                # find the starting and ending coordinates of the big patch
                big_patch_start_x = max(center_x - patch_size_x, min_x)
                big_patch_end_x = min(center_x + patch_size_x, max_x)
                big_patch_start_y = max(center_y - patch_size_y, min_y)
                big_patch_end_y = min(center_y + patch_size_y, max_y)
                big_patch_start_z = max(center_z - patch_size_z, min_z)
                big_patch_end_z = min(center_z + patch_size_z, max_z)
                # if the patch should reach outside the volume, prepare offset for zero padding
                offset_x = max(min_x - (center_x - patch_size_x), 0)
                offset_y = max(min_y - (center_y - patch_size_y), 0)
                offset_z = max(min_z - (center_z - patch_size_z), 0)
                # extract big patch for prediction
                big_patch = np.zeros((1, patch_size_x * 2, patch_size_y * 2, patch_size_z * 2, num_features),
                                     dtype='float32')
                for f in range(num_features):
                    big_patch[0, offset_x:offset_x + (big_patch_end_x - big_patch_start_x),
                              offset_y:offset_y + (big_patch_end_y - big_patch_start_y),
                              offset_z:offset_z + (big_patch_end_z - big_patch_start_z),
                              f] = feature_volumes[f][big_patch_start_x:big_patch_end_x,
                                                      big_patch_start_y:big_patch_end_y,
                                                      big_patch_start_z:big_patch_end_z].astype('float32')

                # normalize patch
                if normalize:
                    assert isinstance(max_values_for_normalization, list)
                    for f in range(num_features):
                        big_patch[..., f] = big_patch[..., f] / max_values_for_normalization[f]

                predicted_patch = model.predict([big_patch, patch], batch_size=1, verbose=0)[-1]

                # in case the last patch along a axis reached outside the volume, cut off the zero padding
                sliced_predicted_patch = predicted_patch[0, :patch_end_x - patch_start_x,
                                                         :patch_end_y - patch_start_y, :patch_end_z - patch_start_z]
                predicted_probabilities[patch_start_x: patch_end_x, patch_start_y: patch_end_y,
                                        patch_start_z: patch_end_z] = sliced_predicted_patch
                p += 1

    # how long does the prediction take for the whole volume
    helper.end_time_measuring(starttime_volume, what_to_measure='volume prediction', print_end_time=False)

    # save annotated volume as nifti
    annotation = np.argmax(predicted_probabilities, axis=-1)
    annotation = np.asarray(annotation, dtype='uint8')
    helper.create_and_save_nifti(annotation, annotation_save_path)

    # wash with segments according to max softmax score in segment and save as nifti
    starttime_washing = helper.start_time_measuring(what_to_measure='washing with segments')
    washed_annotation = helper.wash_with_segments_max_scores(predicted_probabilities, skeleton_segments)
    helper.end_time_measuring(starttime_washing, what_to_measure='washing with segments', print_end_time=False)
    washed_annotation = np.asarray(washed_annotation, dtype='uint8')
    helper.create_and_save_nifti(washed_annotation, washed_annotation_save_path)

    return annotation, washed_annotation


def main(img_filepath, graph_json_filepath, model_filepath, annotation_results_filepath,
         washed_annotation_results_filepath):
    img = helper.load_nifti_mat_from_file(img_filepath)
    graph_json = helper.load_json(graph_json_filepath)
    volume_dimensions = img.shape
    background_value = bravenet_config.ARTER_DICTS[bravenet_config.H_LEVEL]['background']
    num_classes = bravenet_config.NUM_CLASSES
    patch_size_x = bravenet_config.PATCH_SIZE_X
    patch_size_y = bravenet_config.PATCH_SIZE_Y
    patch_size_z = bravenet_config.PATCH_SIZE_Z
    max_values_for_normalization = bravenet_config.MAX_FEATURE_VALUES

    # create volume with vessel skeleton with radius values
    print('Creating skeleton with radius volume...')
    skeleton_radius = helper.skeleton_radius_from_json_graph(graph_json=graph_json, volume_dimensions=volume_dimensions,
                                                             background_value=background_value, radius_dtype='float32')

    # create volume with vessel skeleton with vessel segments coded in y-axis depth
    print('Creating skeleton with vessel segments volume...')
    skeleton_segments, _, _ = helper.skeleton_segments_from_json_graph(graph_json=graph_json,
                                                                       volume_dimensions=volume_dimensions,
                                                                       background_value=background_value,
                                                                       segments_dtype='int32', axis='y')

    feature_volumes = [img, skeleton_radius]
    model = load_model(model_filepath, compile=False)

    predict(feature_volumes=feature_volumes, model=model, num_classes=num_classes, patch_size_x=patch_size_x,
            patch_size_y=patch_size_y, patch_size_z=patch_size_z, annotation_save_path=annotation_results_filepath,
            max_values_for_normalization=max_values_for_normalization, skeleton_segments=skeleton_segments,
            washed_annotation_save_path=washed_annotation_results_filepath)
    return True


if __name__ == '__main__':
    main(img_filepath=sys.argv[1], graph_json_filepath=sys.argv[2], model_filepath=sys.argv[3],
         annotation_results_filepath=sys.argv[4], washed_annotation_results_filepath=sys.argv[5])
