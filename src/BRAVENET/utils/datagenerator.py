"""
This Data generator yields pairs of feature and label samples in model training.
"""
import gzip
import os

import numpy as np
from tensorflow.keras.utils import Sequence

from src.generalutils import helper


class DataGenerator(Sequence):
    def __init__(self, feature_files, feature_files_big, label_files, label_files_big, files_dir, batch_size=32,
                 dim=(32, 32), dim_big=(64, 64), n_channels=3, num_outputs=1, shuffle=False,
                 normalize_features=True, max_values_for_normalization=None):
        """
        :param feature_files: List of file names containing features. List of strings.
        :param feature_files_big: List of file names containing features in double size volume. List of strings.
        :param label_files: List of file names containing labels. List of strings.
        :param label_files_big:List of file names containing labels in double size volume. List of strings.
        :param files_dir: File path to the files directory. String.
        :param batch_size: Size of the loaded batch. Integer.
        :param dim: Feature dimensions. Tuple of integers.
        :param dim_big: Feature dimensions in double size. Tuple of integers.
        :param n_channels: Number of feature channels. Integer.
        :param num_outputs: Number of outputs from the model.
        :param shuffle: True for shuffle the data in every epoch. False otherwise. Boolean.
         matrices. False when the feature channels can be stored in one more dimensional matrix. Boolean.
        :param normalize_features:
        :param max_values_for_normalization:
        """
        self.feature_files = feature_files
        self.feature_files_big = feature_files_big
        self.label_files = label_files
        self.label_files_big = label_files_big
        self.files_dir = files_dir
        self.dim = dim
        self.dim_big = dim_big
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.num_outputs = num_outputs
        self.shuffle = shuffle
        self.normalize_features = normalize_features
        if max_values_for_normalization is not None:
            assert isinstance(max_values_for_normalization, list)
        self.max_values_for_normalization = max_values_for_normalization
        self.X = None  # for normalization sanity check on epoch end
        self.num_samples = len(self.feature_files) if self.feature_files else len(self.feature_files_big)
        self.indexes = np.arange(self.num_samples)

    def __len__(self):
        """
        :return: Number of batches per epoch. Integer.
        """
        return int(np.floor(self.num_samples / float(self.batch_size)))

    def __getitem__(self, index):
        """
        Gets called by fit_generator to retrieve a batch.

        :param index:
        :return: The batch of data (features, labels) for the fit_generator.
        """
        # --- Generate indices of the batch
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # --- Generate data
        X, y = self.__data_generation(indices)
        self.X = X

        return X, y

    def __data_generation(self, indices):
        """
        Loads data features and labels under given indices.

        :param indices: Indices of the data to retrieve.
        :return: The batch of data (features, labels).
        """
        # --- Initialization
        x = [np.empty((self.batch_size, *self.dim_big, self.n_channels), dtype='float32'),
             np.empty((self.batch_size, *self.dim, self.n_channels), dtype='float32')]

        if self.num_outputs > 1:
            y = [np.empty((self.batch_size, *self.dim, 1), dtype='float32')] * self.num_outputs
        else:
            y = np.empty((self.batch_size, *self.dim, 1), dtype='float32')

        # --- Generate data
        for batch_idx, file_idx in enumerate(indices):
            # --- Check that features and label file names match and load data.
            if self.feature_files:
                if not helper.files_match(self.feature_files[file_idx], self.label_files[file_idx],
                                          filename1_split='_features_', filename2_split='_label_'):
                    raise AssertionError('The small feature file name and label file name do not match.')

                assert len(self.feature_files) == len(self.feature_files_big), str(len(
                    self.feature_files)) + ' x ' + str(len(self.feature_files_big))

                if not helper.files_match(self.feature_files[file_idx], self.feature_files_big[file_idx],
                                          filename1_split='size_128', filename2_split='size_256'):
                    raise AssertionError('The feature file name and big feature file name do not match.')

                sample_features = self.__load_volume(self.feature_files[file_idx])
                sample_features_big = self.__load_volume(self.feature_files_big[file_idx])
                sample_label = self.__load_volume(self.label_files[file_idx])
            else:
                if not helper.files_match(self.feature_files_big[file_idx], self.label_files_big[file_idx]):
                    raise AssertionError('The feature file name and label file name do not match.')

                sample_features = self.__load_volume(self.feature_files_big[file_idx], cut_out=True)
                sample_features_big = self.__load_volume(self.feature_files_big[file_idx])
                sample_label = self.__load_volume(self.label_files_big[file_idx], cut_out=True)
            assert sample_features.shape[-1] == self.n_channels

            if self.normalize_features:
                for channel in range(self.n_channels):
                    sample_features[..., channel] = sample_features[..., channel] / self.max_values_for_normalization[
                        channel]
                    sample_features_big[..., channel] = sample_features_big[..., channel] / \
                                                        self.max_values_for_normalization[channel]

            # --- Store sample features
            x[0][batch_idx] = sample_features_big
            x[1][batch_idx] = sample_features

            # --- Store sample label
            if self.num_outputs > 1:
                for i in range(self.num_outputs):
                    y[i][batch_idx] = np.expand_dims(sample_label, axis=-1)
            else:
                y[batch_idx] = np.expand_dims(sample_label, axis=-1)

        return x, y

    def __load_volume(self, filename, cut_out=False):
        """
        Specifies volume loading.

        :param filename: File containing the desired volume to be loaded.
        :return: Data content.
        """
        filepath = os.path.join(self.files_dir, filename)
        matrix = None
        if filename.endswith('npy.gz'):
            unzipped_file = gzip.GzipFile(filepath, "r")
            matrix = np.load(unzipped_file)
        elif filename.endswith('npz'):
            matrix = np.load(filepath)['arr_0']
        elif filename.endswith('npy'):
            matrix = np.load(filepath)
        else:
            ValueError('Unknown data format of the patch files.')
        if cut_out:
            matrix = helper.half_center_matrix(matrix, self.dim_big)
        return matrix

    def on_epoch_end(self):
        """
        Gets called after __len__ batches have been retrieved.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

        return
