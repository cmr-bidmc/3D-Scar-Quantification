from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#
# This function calculates Dice scores and implements various loss functions (Cross-entropy, Dice, Jaccard)
# (loss function inspired by: https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/loss.py)
#

# Calculate Dice coefficient from labels and output probabilities
def dice_coef(output_probs, labels_flat):
    with tf.variable_scope("max_preds"):
        num_labels_output = labels_flat.get_shape()[1].value

    with tf.variable_scope("dice_calc"):
        epsilon = 1e-3
        # NOTE: Dice is calculated with softmax probabilities and NOT binary (0,1) thresholded segmentation predictions, because hard thresholding has 0 gradient
        mult = tf.multiply(output_probs, labels_flat)
        intersection = 2 * tf.reduce_sum(mult, 0)
        intersection = tf.add(intersection, epsilon)
        denominator = tf.add(tf.reduce_sum(output_probs, 0), tf.reduce_sum(labels_flat, 0))
        denominator = tf.add(denominator, epsilon)
        dice_coef_per_class = tf.truediv(intersection, denominator)

    with tf.variable_scope("true_dice"):
        class_of_max_prediction = tf.argmax(output_probs, axis=1)
        binary_per_class_prediction = tf.one_hot(class_of_max_prediction, num_labels_output)
        _mult = tf.multiply(binary_per_class_prediction, labels_flat)
        _intersection = 2 * tf.reduce_sum(_mult, 0)
        _intersection = tf.add(_intersection, epsilon)
        _denominator = tf.add(tf.reduce_sum(binary_per_class_prediction, 0), tf.reduce_sum(labels_flat, 0))
        _denominator = tf.add(_denominator, epsilon)
        true_dice_coef_per_class = tf.truediv(_intersection, _denominator)

        return dice_coef_per_class, true_dice_coef_per_class, class_of_max_prediction

# Calculate Jaccard index from Dice index
def jaccard_index(dice_coef_per_class):
    return tf.truediv(dice_coef_per_class, 2 - dice_coef_per_class)


# Calculate the loss from the logits and the labels
def loss(logits, labels, hparams, head=None):

    # Get number of output labels from hyperparameters
    num_classes = hparams.num_labels_output

    # Get type of loss to be used for network optimization
    loss_type = hparams.loss_type

    # Define weights for each class
    num_of_background_voxels = 5556947
    num_of_bloodpool_voxels = 4343036
    num_of_wall_voxels = 1822033
    num_of_plaque_voxels = 74464
    #ratio = num_of_plaque_voxels / float(num_of_background_voxels + num_of_bloodpool_voxels + num_of_wall_voxels + num_of_plaque_voxels)
    ratio = 0.000000001
    class_weights = tf.constant([ratio, ratio, ratio, 1-ratio])

    # Reshape logits to compare values of each voxel
    print("Shapes before reshape: ")
    print(labels.shape)
    print(logits.shape)

    logits = tf.reshape(logits, (-1, num_classes))
       # Apply class weighting to logits if specified in hyperparameters
    if hparams.use_class_weighting is True: # Ahmed: I believe this is wrong. It should be used to weight the error
        print("Using class weights on logits ...")
        logits = tf.multiply(logits, class_weights)
        print("Use following class weight ratio for weighted logits: " + str(ratio))
        print("Shape of weighted logits: " + str(logits))

    # Reshape labels to compare values of each voxel
    epsilon = tf.constant(value=1e-4)
    labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

    # Generate output probabilities from weighted logits
    softmax = tf.nn.softmax(logits) + epsilon

    # Get current Dice score
    with tf.variable_scope("dice"):
        dice_coef_per_class, true_dice_coef_per_class, class_of_max_prediction = dice_coef(softmax, labels)

    # Get current Jaccard score from Dice score
    with tf.variable_scope("dice"):
        true_jaccard_index_per_class = jaccard_index(true_dice_coef_per_class)

    # Calculate loss values
    with tf.name_scope('loss'):

        # Calculate the actual cross entropy from labels and softmax
        if loss_type == 'cross_entropy':
            print("Shapes after reshape: ")
            print(labels.shape)
            print(softmax.shape)
            if head is not None:
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), head), axis=[1])
            else:
                cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), axis=[1])

            # Get mean cross entropy value over all voxels
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            loss = cross_entropy_mean
            if hparams.balance_scar_size is True:
                loss = loss * tf.log(10+tf.reduce_sum(labels[:,3]))
            print("Shape of loss: ")
            print(loss.shape)

        # Convert to and use Dice coefficient as loss
        elif loss_type == 'dice':
            dice_coeffs_loss = tf.subtract(1.0, dice_coef_per_class)
            dice_loss = tf.reduce_mean(dice_coeffs_loss[1:], name='dice_loss')
            loss = dice_loss

        # Convert to and use class-weighted Dice coefficient as loss
        elif loss_type == 'weighted_dice':
            dice_coeffs_loss = tf.subtract(1.0, dice_coef_per_class)
            dice_coeffs_class_weighted_loss = [1, 1, 2, 50] * dice_coeffs_loss
            dice_loss = tf.reduce_mean(dice_coeffs_class_weighted_loss[1:], name='weighted_dice_loss')
            loss = dice_loss

        # Convert to and use class-weighted Dice coefficient as loss with different value to loss conversion
        elif loss_type == 'weighted_dice_2':
            dice_coeffs_loss = tf.truediv(1.0, dice_coef_per_class)
            dice_coeffs_class_weighted_loss = [1, 1, 2, 10] * dice_coeffs_loss
            dice_loss_2 = tf.reduce_mean(dice_coeffs_class_weighted_loss[1:], name='weighted_dice_loss_2')
            loss = dice_loss_2

        # Calculate Jaccard coefficient from Dice scores and use it as loss
        elif loss_type == 'jaccard':
            jaccard_index_per_class = jaccard_index(dice_coef_per_class)
            jaccard_coeffs_loss = tf.subtract(1.0, jaccard_index_per_class)
            jaccard_loss = tf.reduce_mean(jaccard_coeffs_loss[1:], name='jaccard_loss')
            loss = jaccard_loss

    return loss, dice_coef_per_class, true_dice_coef_per_class
