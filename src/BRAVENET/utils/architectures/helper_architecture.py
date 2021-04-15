"""
This file contains helper functions for the scripts defining the network architecture.
"""

import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Convolution3D, Activation, Conv3DTranspose, UpSampling3D, \
    Dropout


def convolutional_block(input_layer, n_filters, activation=None, filter_size=(3, 3, 3), strides=(1, 1, 1),
                        padding='same', data_format='channels_last', batch_normalization=False, dropout=None,
                        l1=None, l2=None, axis=4, use_bias=True, bn_name=None, conv_name=None):
    """
    Block of one convolutional layer with possible activation and batch normalization.

    :param input_layer: Input layer to the convolution. Keras layer.
    :param n_filters: Number of filters in the particular convolutional layer. Positive integer.
    :param activation: Activation_function after every convolution. String activation name.
    :param filter_size: Size of the convolution filter (kernel). Tuple of 3 positive integers.
    :param strides: Strides values. Tuple of 3 positive integers.
    :param padding: Used padding by convolution. Takes values: 'same' or 'valid'.
    :param data_format: Ordering of the dimensions in the inputs. Takes values: 'channel_first' or 'channel_last'.
    :param batch_normalization: If set to True, will use Batch Normalization layers after each convolution layer.
    :param dropout: percentage of weights to be dropped, float between 0 and 1.
    :param l1: L1 regularization.
    :param l2: L2 regularization.
    :param axis: Axis for batch normalization.
    :param use_bias:
    :param bn_name: Name of the batch normalization layer. String.
    :param conv_name:
    :return: Keras layer.
    """

    if l1 is not None:
        if l2 is not None:
            layer = Convolution3D(filters=n_filters, kernel_size=filter_size, strides=strides, padding=padding,
                                  data_format=data_format, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                  use_bias=use_bias, name=conv_name)(input_layer)
        else:
            layer = Convolution3D(filters=n_filters, kernel_size=filter_size, strides=strides, padding=padding,
                                  data_format=data_format, kernel_regularizer=regularizers.l1(l1),
                                  use_bias=use_bias, name=conv_name)(input_layer)
    else:
        if l2 is not None:
            layer = Convolution3D(filters=n_filters, kernel_size=filter_size, strides=strides, padding=padding,
                                  data_format=data_format, kernel_regularizer=regularizers.l2(l2), use_bias=use_bias,
                                  name=conv_name)(
                input_layer)
        else:
            layer = Convolution3D(filters=n_filters, kernel_size=filter_size, strides=strides, padding=padding,
                                  data_format=data_format, use_bias=use_bias, name=conv_name)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=axis, name=bn_name)(
            layer)  # the axis that should be normalized (typically the features axis), integer.
        # Layer input shape: (samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'
    if activation is not None:
        layer = Activation(activation=activation)(layer)
    if dropout is not None:
        layer = Dropout(dropout)(layer)
    return layer


def upscaling(input_layer, n_filters, pool_size=(2, 2, 2), padding='same', data_format='channels_last',
              deconvolution=False):
    """
    Up-scaling layer. It can be either transpose convolutional layer (deconvolution) or up-sampling layer.

    :param input_layer: Input layer to the convolution. Keras layer.
    :param n_filters: Number of filters in the deconvolution. Positive integer.
    :param pool_size: Factors by which to downscale (height, width, depth). Tuple of 3 positive integers.
    :param padding: Used padding by convolution. Takes values: 'same' or 'valid'.
    :param data_format: Ordering of the dimensions in the inputs. Takes values: 'channel_first' or 'channel_last'.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Keras layer.
    """

    if deconvolution:
        return Conv3DTranspose(filters=n_filters, kernel_size=pool_size, strides=pool_size, padding=padding,
                               data_format=data_format)(input_layer)
    else:
        return UpSampling3D(size=pool_size, data_format=data_format)(input_layer)


def ds_loss(depth=4, loss_function='categorical_crossentropy', layer_name='out-', start_offset=0):
    """
    Loss for deep supervision.

    :param depth: Positive int. Network depth.
    :param loss_function: String. Loss function.
    :param layer_name: String. Name prefix of output layers in levels.
    :param start_offset: Int.
    :return: Dictionary of loss functions for each level. Dictionary of loss weights for each level.
    """
    loss = {}
    loss_weights = {}
    for i in range(depth - 1):
        loss[layer_name + str(i + start_offset)] = loss_function
        loss_weights[layer_name + str(i + start_offset)] = 0.5 if i == 0 else 0.5 / (depth - 2)
    return loss, loss_weights
