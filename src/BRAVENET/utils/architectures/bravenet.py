"""
This file defines the brave-net architecture.
"""

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, concatenate, BatchNormalization, MaxPooling3D, Convolution3D, \
    AveragePooling3D, UpSampling3D
from tensorflow.keras.models import Model

from src.BRAVENET.utils.architectures.helper_architecture import upscaling, convolutional_block, ds_loss


def get_bravenet(input_shapes, n_classes=1, activation='relu', final_activation='softmax', optimizer=None,
                 learning_rate=1e-4, n_base_filters=32, depth=3, dropout=None, l1=None, l2=None,
                 loss_function=None,
                 metrics='accuracy', filter_size=(3, 3, 3), pool_size=(2, 2, 2), concat_axis=-1,
                 data_format='channels_last',
                 padding='same', batch_normalization=False, deconvolution=False):
    """
    Defines the architecture of the brave-net.

    :param input_shapes: List of shapes of the input data. List of tuples of 4 positive integers (x_size, y_size,
    z_size, n_channels).
    The x, y, and z sizes must be divisible by the pool size to the power of the depth of the net, that is
    pool_size^depth.
    :param n_classes: Number of class labels that the model is learning. Positive integer.
    :param activation: Activation_function after every convolution. String activation name.
    :param final_activation: Activation_function of the final layer. String activation name.
    :param optimizer: Optimization algorithm for updating the weights and bias values.
    :param learning_rate: Learning_rate of the optimizer. Float.
    :param dropout: Percentage of weights to be dropped. Float between 0 and 1.
    :param l1: L1 regularization.
    :param l2: L2 regularization.
    :param n_base_filters: The number of filters that the first layer in the network will have. Positive integer.
    Following layers will contain a multiple of this number. Lowering this number will likely reduce the amount of
    memory required
    to train the model.
    :param depth: Number of network's levels. Positive integer.
    :param loss_function: Loss function also known as cost function. String function name.
    :param metrics: Metrics for evaluation of the model performance. List of metrics names.
    :param filter_size: sSze of the convolution kernel. Tuple of 3 positive integers.
    :param pool_size: Factors by which to downscale (height, width, depth). Tuple of 3 positive integers.
    :param concat_axis: Concatenation axis, concatenate over channels. Positive integer.
    :param data_format: Ordering of the dimensions in the inputs. Takes values: 'channel_first' or 'channel_last'.
    :param padding: Used padding by convolution. Takes values: 'same' or 'valid'.
    :param batch_normalization: If set to True, will use Batch Normalization layers after each convolution layer
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling.
    This increases the amount memory required during training.
    :return: Compiled untrained 3D brave-net model.
    """

    ### DOWN-SCALE PATHS
    model_inputs = []
    down_path_outputs = []
    up_path_outputs = []
    # store the output of the 2. convolution in the downsampling path to be able to load it in the upsampling path
    # for concatenation
    down_path_residuals_list = []

    # Get low resolution patch size
    lri = not input_shapes[0][0] > input_shapes[1][0]

    ### BUILD SEPARATE DOWN-SAMPLING PATH FOR EACH INPUT
    for i, input_shape in enumerate(input_shapes):
        # specify the input shape
        model_inputs.append(Input(input_shape, name='in-' + str(i)))
        current_layer = model_inputs[-1]
        residuals = []

        # BN for inputs
        if batch_normalization:
            current_layer = BatchNormalization(axis=concat_axis, name='bn-' + str(i))(current_layer)

        # scale down input to context path
        if i == lri:
            current_layer = AveragePooling3D(pool_size=pool_size)(current_layer)

        # build down-sampling path (left side of the brave-net)
        # layers on each level: convolution3d -> BN -> convolution3d -> BN -> max-pooling
        # last level without max-pooling
        for level in range(depth):
            conv1 = convolutional_block(input_layer=current_layer, n_filters=n_base_filters * (2 ** level),
                                        activation=activation, filter_size=filter_size, padding=padding,
                                        data_format=data_format, batch_normalization=batch_normalization,
                                        dropout=dropout, axis=concat_axis,
                                        bn_name='bn-down-input-' + str(i) + '-level-' + str(level) + '-0',
                                        conv_name='conv-down-input-' + str(i) + '-level-' + str(level) + '-0')
            conv2 = convolutional_block(input_layer=conv1, n_filters=n_base_filters * (2 ** level) * 2,
                                        activation=activation, filter_size=filter_size, padding=padding,
                                        data_format=data_format, batch_normalization=batch_normalization,
                                        dropout=None, axis=concat_axis,
                                        bn_name='bn-down-input-' + str(i) + '-level-' + str(level) + '-1',
                                        conv_name='conv-down-input-' + str(i) + '-level-' + str(level) + '-1')
            residuals.append(conv2)
            if level < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size, data_format=data_format)(conv2)
            else:
                current_layer = conv2
        down_path_outputs.append(current_layer)
        down_path_residuals_list.append(residuals)

    ### BOTTLENECK
    # Concatenate feature maps
    bottleneck_concat = concatenate(down_path_outputs, axis=concat_axis)
    # Fully connected 1
    fl1 = Convolution3D(filters=n_base_filters * (2 ** (depth - 1)) * 2, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                        padding=padding, data_format=data_format, activation=activation,
                        name='conv-fl-0')(bottleneck_concat)
    # Fully connected 2
    fl2 = Convolution3D(filters=n_base_filters * (2 ** (depth - 1)) * 2, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                        padding=padding, data_format=data_format, activation=activation,
                        name='conv-fl-1')(fl1)

    ### UP-SAMPLING PART (right side of the brave-net)
    # layers on each level: upconvolution3d -> concatenation with feature maps of corresponding level from down-sampling
    # part -> convolution3d -> BN -> convolution3d -> BN
    current_layer = fl2
    for level in range(depth - 2, -1, -1):
        up_scale = upscaling(input_layer=current_layer, n_filters=current_layer.shape[-1],
                             pool_size=pool_size, padding=padding, data_format=data_format,
                             deconvolution=deconvolution)  # _keras_shape: output layer shape
        # (samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if data_format='channels_last'
        merged_residuals = concatenate([down_path_residuals_list[0][level], down_path_residuals_list[1][level]],
                                       name='concat-' + str(level) + '-0')
        concat = concatenate([merged_residuals, up_scale], axis=concat_axis,
                             name='concat-' + str(level) + '-1')
        current_layer = convolutional_block(input_layer=concat,
                                            n_filters=down_path_residuals_list[0][level].shape[-1],
                                            activation=activation, filter_size=filter_size, padding=padding,
                                            data_format=data_format, batch_normalization=batch_normalization,
                                            dropout=dropout, axis=concat_axis,
                                            bn_name='bn-up-level-' + str(level) + '-0',
                                            conv_name='conv-up-level-' + str(level) + '-0')
        current_layer = convolutional_block(input_layer=current_layer,
                                            n_filters=down_path_residuals_list[0][level].shape[-1],
                                            activation=activation, filter_size=filter_size, padding=padding,
                                            data_format=data_format, batch_normalization=batch_normalization,
                                            dropout=None, axis=concat_axis,
                                            bn_name='bn-up-level-' + str(level) + '-1',
                                            conv_name='conv-up-level-' + str(level) + '-1')
        if 0 < level < depth - 1:
            # get level prediction
            output = UpSampling3D(size=(int(input_shapes[1][0] / current_layer.shape[1]),
                                        int(input_shapes[1][1] / current_layer.shape[2]),
                                        int(input_shapes[1][2] / current_layer.shape[3])),
                                  data_format=data_format)(current_layer)
            output = Convolution3D(filters=n_classes, kernel_size=1, activation=final_activation, padding=padding,
                                   data_format=data_format, name='out-' + str(level))(output)
            up_path_outputs.append(output)

    # final convolutional layer maps feature maps to desired number of classes
    if l1 is not None:
        if l2 is not None:
            final_conv = Convolution3D(filters=n_classes, kernel_size=1, activation=final_activation, padding=padding,
                                       data_format=data_format, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                       name='out-0')(
                current_layer)
        else:
            final_conv = Convolution3D(filters=n_classes, kernel_size=1, activation=final_activation, padding=padding,
                                       data_format=data_format, kernel_regularizer=regularizers.l1(l1),
                                       name='out-0')(current_layer)
    else:
        if l2 is not None:
            final_conv = Convolution3D(filters=n_classes, kernel_size=1, activation=final_activation, padding=padding,
                                       data_format=data_format, kernel_regularizer=regularizers.l2(l2),
                                       name='out-0')(
                current_layer)
        else:
            final_conv = Convolution3D(filters=n_classes, kernel_size=1, activation=final_activation, padding=padding,
                                       data_format=data_format,
                                       name='out-0')(current_layer)
    up_path_outputs.append(final_conv)

    # create the model
    model = Model(inputs=model_inputs, outputs=up_path_outputs, name='bravenet')

    # configure the learning process via the compile function
    if not isinstance(metrics, list):
        metrics = [metrics]
    # Multiple loss
    loss, loss_weights = ds_loss(depth=depth, loss_function=loss_function)
    model.compile(optimizer=optimizer(lr=learning_rate), loss=loss, metrics=metrics, loss_weights=loss_weights)
    print('brave-net compiled.')

    # print out model summary to console
    model.summary(line_length=120)
    return model
