from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from math import ceil
import numpy as np
import tensorflow as tf

def deep_model_128in(features, hparams):
    phase = 'train'
    weight_decay = 5e-4
    conv1 = _conv_layer_no_relu(bottom=features, name="cnv1", output_channels=64, phase=phase,
                                weight_decay=weight_decay)

    bottleneck1_1 = _bottleneck_block(bottom=conv1, name="bn1_1", output_channels=64, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck1_2 = _bottleneck_block(bottom=bottleneck1_1, name="bn1_2", output_channels=64, downsample=True,
                                      train=True, phase=phase, weight_decay=weight_decay)

    bottleneck1_3 = _bottleneck_block(bottom=bottleneck1_2, name="bn1_3", output_channels=64,
                                      train=True, phase=phase, weight_decay=weight_decay)
    bottleneck1_4 = _bottleneck_block(bottom=bottleneck1_3, name="bn1_4", output_channels=64,
                                      train=True, phase=phase, weight_decay=weight_decay)
    bottleneck1_5 = _bottleneck_block(bottom=bottleneck1_4, name="bn1_5", output_channels=64,
                                      train=True, phase=phase, weight_decay=weight_decay)


    bottleneck2_1 = _bottleneck_block(bottom=bottleneck1_5, name="bn2_1", output_channels=128, downsample=True,
                                      train=True, phase=phase, weight_decay=weight_decay)
    bottleneck2_2 = _bottleneck_block(bottom=bottleneck2_1, name="bn2_2", output_channels=128, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck2_3 = _bottleneck_block(bottom=bottleneck2_2, name="bn2_3", output_channels=128, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck2_4 = _bottleneck_block(bottom=bottleneck2_3, name="bn2_4", output_channels=128, train=True, phase=phase,
                                      weight_decay=weight_decay)

    bottleneck3_1 = _bottleneck_block(bottom=bottleneck2_4, name="bn3_1", output_channels=256, downsample=True,
                                      train=True, phase=phase, weight_decay=weight_decay)
    bottleneck3_2 = _bottleneck_block(bottom=bottleneck3_1, name="bn3_2", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck3_3 = _bottleneck_block(bottom=bottleneck3_2, name="bn3_3", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck3_4 = _bottleneck_block(bottom=bottleneck3_3, name="bn3_4", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck3_5 = _bottleneck_block(bottom=bottleneck3_4, name="bn3_5", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck3_6 = _bottleneck_block(bottom=bottleneck3_5, name="bn3_6", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck3_7 = _bottleneck_block(bottom=bottleneck3_6, name="bn3_7", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck3_8 = _bottleneck_block(bottom=bottleneck3_7, name="bn3_8", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)



    bottleneck4_1 = _bottleneck_block(bottom=bottleneck3_8, name="bn4_1", output_channels=512, train=True,
                                      phase=phase, weight_decay=weight_decay)
    bottleneck4_2 = _bottleneck_block(bottom=bottleneck4_1, name="bn4_2", output_channels=512, train=True,
                                      phase=phase, weight_decay=weight_decay)
    bottleneck4_3 = _bottleneck_block(bottom=bottleneck4_2, name="bn4_3", output_channels=512, train=True,
                                      phase=phase, weight_decay=weight_decay)


    bottleneck5_1 = _bottleneck_block(bottom=bottleneck4_3, name="bn5_1", output_channels=256, upsample=True,
                                      train=True, phase=phase, weight_decay=weight_decay)
    long_skip5 = _res_layer(bottleneck2_3, bottleneck5_1, 'long_skip5', phase=phase, weight_decay=weight_decay,
                            merger_concat=True)
    bottleneck5_2 = _bottleneck_block(bottom=long_skip5, name="bn5_2", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck5_3 = _bottleneck_block(bottom=bottleneck5_2, name="bn5_3", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck5_4 = _bottleneck_block(bottom=bottleneck5_3, name="bn5_4", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck5_5 = _bottleneck_block(bottom=bottleneck5_4, name="bn5_5", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck5_6 = _bottleneck_block(bottom=bottleneck5_5, name="bn5_6", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck5_7 = _bottleneck_block(bottom=bottleneck5_6, name="bn5_7", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck5_8 = _bottleneck_block(bottom=bottleneck5_7, name="bn5_8", output_channels=256, train=True, phase=phase,
                                      weight_decay=weight_decay)

    bottleneck6_1 = _bottleneck_block(bottom=bottleneck5_8, name="bn6_1", output_channels=64, upsample=True,
                                      train=True, phase=phase, weight_decay=weight_decay)
    long_skip6 = _res_layer(bottleneck1_5, bottleneck6_1, 'long_skip6', phase=phase, merger_concat=True)
    bottleneck6_2 = _bottleneck_block(bottom=long_skip6, name="bn6_2", output_channels=64, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck6_3 = _bottleneck_block(bottom=bottleneck6_2, name="bn6_3", output_channels=64, train=True, phase=phase,
                                      weight_decay=weight_decay)
    bottleneck6_4 = _bottleneck_block(bottom=bottleneck6_3, name="bn6_4", output_channels=64, train=True, phase=phase,
                                      weight_decay=weight_decay)

    bottleneck7_1 = _bottleneck_block(bottom=bottleneck6_4, name="bn7_1", output_channels=64, train=True,
                                      upsample=True, phase=phase, weight_decay=weight_decay)
    long_skip7 = _res_layer(conv1, bottleneck7_1, 'long_skip7', phase=phase, merger_concat=True)
    bottleneck7_2 = _bottleneck_block(bottom=long_skip7, name="bn7_2", output_channels=64, train=True, phase=phase,
                                      weight_decay=weight_decay)

    conv7 = _conv_layer_no_relu(bottom=bottleneck7_2, name="conv7", output_channels=64, phase=phase,
                                weight_decay=weight_decay)

    bottleneck_skip1 = _bottleneck_block(bottom=features, name="bn_sk1", output_channels=64, train=True, phase=phase,
                                         weight_decay=weight_decay)
    bottleneck_skip2 = _bottleneck_block(bottom=bottleneck_skip1, name="bn_sk2", output_channels=64, train=True,
                                         phase=phase, weight_decay=weight_decay)
    bottleneck_skip3 = _bottleneck_block(bottom=bottleneck_skip2, name="bn_sk3", output_channels=64, train=True,
                                         phase=phase, weight_decay=weight_decay)
    bottleneck_skip4 = _bottleneck_block(bottom=bottleneck_skip3, name="bn_sk4", output_channels=64, train=True,
                                         phase=phase, weight_decay=weight_decay)

    long_skip8 = _res_layer(bottleneck_skip4, conv7, 'long_skip8', phase=phase, merger_concat=True)

    logits = _prediction_block(long_skip8, 'prediction_block', True, num_labels_output=hparams.num_labels_output,
                               phase=phase, weight_decay=weight_decay)

    softmax = tf.nn.softmax(logits)
    return logits, softmax

def _prediction_block(bottom, name, train=True, debug=True, num_labels_output=2, phase='', weight_decay=5e-4):
    with tf.variable_scope(name) as scope:
        input_channels = bottom.get_shape()[3].value
        phase_train = tf.Variable(train, name='phase_train', trainable=False)

        normed1 = tf.contrib.layers.batch_norm(bottom, is_training=train, updates_collections=None)
        relu1 = tf.nn.relu(normed1, 'relu1')
        conv1 = _conv_layer_no_relu(relu1, 'conv1', num_labels_output, ksize=[1, 1, 1], phase=phase,
                                    weight_decay=weight_decay)
        return conv1

def _bottleneck_block(bottom, name, output_channels, downsample=False, upsample=False, train=True, debug=True, phase='',
                      weight_decay=5e-4):
    with tf.variable_scope(name) as scope:

        input_channels = bottom.get_shape()[4].value
        res_input = bottom
        phase_train = tf.Variable(train, name='phase_train', trainable=False)

        normed1 = tf.contrib.layers.batch_norm(bottom, is_training=train, updates_collections=None)
        relu1 = tf.nn.relu(normed1, 'relu1')
        if downsample == True:
            conv1 = _conv_layer_no_relu(relu1, 'cnv1', output_channels / 4, ksize=[1, 1, 1], strides=[1, 2, 2, 1, 1],
                                        phase=phase, weight_decay=weight_decay)
            res_input = _conv_layer_no_relu(res_input, 'inpt_dwn', input_channels, ksize=[1, 1, 1], strides=[1, 2, 2, 1, 1],
                                            trainable=False, phase=phase, weight_decay=weight_decay)
        else:
            conv1 = _conv_layer_no_relu(relu1, 'cnv1', output_channels / 4, ksize=[1, 1, 1], phase=phase,
                                        weight_decay=weight_decay)

        normed2 = tf.contrib.layers.batch_norm(conv1, is_training=train, updates_collections=None)
        relu2 = tf.nn.relu(normed2, 'relu2')
        conv2 = _conv_layer_no_relu(relu2, 'cnv2', output_channels / 4, ksize=[3, 3, 3], phase=phase,
                                    weight_decay=weight_decay)

        normed3 = tf.contrib.layers.batch_norm(conv2, is_training=train, updates_collections=None)
        relu3 = tf.nn.relu(normed3, 'relu2')
        if upsample == True:
            print("Upsampling step ...")
            print("Shape before relu3_up: " + str(relu3.get_shape()))
            relu3_up = _upconv_layer(relu3, name="up", output_channels=int(output_channels / 4),
                                     weight_decay=weight_decay)
            print("Shape after relu3_up: " + str(relu3_up.get_shape()))
            conv3 = _conv_layer_no_relu(relu3_up, 'cnv3', output_channels, ksize=[1, 1, 1], phase=phase,
                                        weight_decay=weight_decay)
            print("Shape before res_input: " + str(res_input.get_shape()))
            res_input = _upconv_layer(res_input, name='inpt_up', output_channels=input_channels, trainable=False,
                                      weight_decay=weight_decay)
            print("Shape after res_input: " + str(res_input.get_shape()))
        else:
            conv3 = _conv_layer_no_relu(relu3, 'cnv3', output_channels, ksize=[1, 1, 1], phase=phase,
                                        weight_decay=weight_decay)

        if phase == 'train':
            conv3 = tf.nn.dropout(conv3, 0.5)

        res_connection = _res_layer(res_input, conv3, 'res', weight_decay)
        print("Leaving this bottleneck block with shape of res_connection: " + str(res_connection.get_shape()))
        return res_connection


def crop_and_concat(x1, x2):  # ASF
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


def _res_layer(bottom1, bottom2, name, debug=True, phase='', weight_decay=5e-4, merger_concat=False):
    with tf.variable_scope(name) as scope:
        print("Entering res_layer")
        print(bottom1.get_shape())
        print(bottom2.get_shape())
        if bottom1.get_shape()[4].value != bottom2.get_shape()[4].value:
            bottom1 = _conv_layer_no_relu(bottom1, 'chan_num', bottom2.get_shape()[4].value, ksize=[1, 1, 1], phase=phase)

        if merger_concat is False:
            added = tf.add(bottom1, bottom2)
        else:
            added = tf.concat([bottom1, bottom2], 4)

        return added


def _conv_layer_no_relu(bottom, name, output_channels=None, ksize=[3, 3, 3], strides=[1, 1, 1, 1, 1], padding='SAME',
                        debug=True, trainable=True, phase='', weight_decay=5e-4):
    with tf.variable_scope(name) as scope:
        n = bottom.get_shape()[4].value
        if output_channels is None:
            output_channels = n
        shape = [ksize[0], ksize[1], ksize[2], n, output_channels]
        num_input = ksize[0] * ksize[1] * ksize[2] * n
        stddev = (2 / num_input) ** 0.5
        weights = _weight_variable(shape, stddev, trainable, phase, weight_decay)
        bias = _bias_variable([output_channels], constant=0.0, trainable=trainable, phase=phase)
        conv = tf.nn.conv3d(bottom, weights,
                            strides=strides, padding=padding)

        bias_layer = tf.nn.bias_add(conv, bias, name=scope.name)
        return bias_layer


def _upconv_layer(bottom,
                  name, output_channels, debug=True, shape=None,
                  ksize=4, stride=2, trainable=True, phase='', weight_decay=5e-4):
    print("Performing upconvolution ...")

    strides = [1, stride, stride, 1, 1]

    with tf.variable_scope(name):
        in_features = bottom.get_shape()[4].value

        if shape is None:
            # Compute shape out of Bottom
            batch_size = bottom.get_shape()[0].value
            d = (bottom.get_shape()[1].value * stride)
            h = (bottom.get_shape()[2].value * stride)
            w = (bottom.get_shape()[3].value * 1)
            new_shape = [batch_size, d, h, w, output_channels]
        else:
            new_shape = [shape[0], shape[1], shape[2], shape[3], output_channels]
        output_shape = tf.stack(new_shape)


        f_shape = [ksize, ksize, ksize, output_channels, in_features]

        weights = _get_deconv_filter(f_shape, trainable, weight_decay)

        print("input tensor shape : " + str(bottom.get_shape()))
        print("weights tensor shape : " + str(weights.get_shape()))
        print("output shape : " + str(output_shape))
        deconv = tf.nn.conv3d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
        print("tensor shape after deconv op : " + str(deconv.get_shape()))
    return deconv


def _get_deconv_filter(f_shape, trainable=True, weight_decay=5e-4):
    depth = f_shape[0]
    height = f_shape[1]
    width = f_shape[2]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1], f_shape[2]])
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c)) * (1 - abs(z / f - c))
                bilinear[x, y, z] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[3]):
        weights[:, :, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    var = tf.get_variable(name="up_fltr", initializer=init,
                          shape=weights.shape, trainable=trainable)

    return var


def _bias_variable(shape, constant=0.0, trainable=True, phase=''):
    initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name='b', shape=shape,
                          initializer=initializer, trainable=trainable)
    return var


def _weight_variable(shape, stddev=0.01, trainable=True, phase='', weight_decay=5e-4):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name='W', shape=shape,
                          initializer=initializer, trainable=trainable)
    return var
