import tensorflow as tf
import os
import numpy as np
import time
import scipy
import hyperparameters
from model_3D import deep_model_128in as model
from loss import loss
from data_input import load_data

from helper_files.data3d_augment_reformat import augment_n_reformat

# Get hyperparameters
hparams = hyperparameters.create_hparams()

# Set data input path
train_data_dir = hparams.input_dir +'/train'

# Set path to save images to during training
current_results_dir_base = os.path.join(hparams.results_dir, 'during_training')
images_output_path = os.path.join(current_results_dir_base, hparams.model_subdir)

# Load preprocessed data
images, labels = load_data(train_data_dir)
# For the first epoch, do not do data augmentation
images_aug, labels_aug = augment_n_reformat(images, labels, hparams)

# Set up input data placeholder variables
x = tf.placeholder(dtype = tf.float32, shape=images_aug.shape)
y = tf.placeholder(dtype = tf.float32, shape=labels_aug.shape)
input_images = tf.Variable(x, trainable=False, collections=[])
input_labels = tf.Variable(y, trainable=False, collections=[])
image, label = tf.train.slice_input_producer([input_images, input_labels])
images_batch, labels_batch = tf.train.batch([image, label], batch_size=hparams.batch_size)

# Feed input data batch into neural network architecture
logits, _ = model(images_batch, hparams)

# Calculate loss of current network
total_loss, dice_coef_per_class, true_dice_coef_per_class = loss(logits, labels_batch, hparams)

# Attach summary operations to tensors to visualize training progress
tf.summary.scalar('loss', total_loss)
tf.summary.scalar('dice_background', true_dice_coef_per_class[0])
tf.summary.scalar('dice_bloodpool', true_dice_coef_per_class[1])
tf.summary.scalar('dice_wall', true_dice_coef_per_class[2])
tf.summary.scalar('dice_plaque', true_dice_coef_per_class[3])

# Merge all summaries to force them to get actual values
summary_op = tf.summary.merge_all()

# Set up counter variable for number of training steps
global_step = tf.Variable(0, name='global_step', trainable=False)

# Define training operation/optimizer to be used
train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(total_loss, global_step=global_step)

# Train/Run the network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # Run the operations to initialize the variables
    sess.run(tf.global_variables_initializer())
    sess.run(input_images.initializer,
             feed_dict={x: images_aug})
    sess.run(input_labels.initializer,
             feed_dict={y: labels_aug})

    # Start input enqueue threads to fill input queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create a tensorflow saver to save model architecture and variables values for later restoration
    saver = tf.train.Saver(max_to_keep=None)

    # If corresponding hyperparameter is set, restore model at given checkpoint
    if hparams.restore_model == True and len(hparams.model_subdir) > 0:

        # Get checkpoint file to restore model from
        checkpoint_dir = os.path.join(os.path.abspath((hparams.model_dir)), hparams.model_subdir)
        checkpoint_filename = os.path.join(checkpoint_dir, 'model.ckpt-' + str(hparams.checkpoint_number))
        meta_checkpoint_filename = os.path.join(checkpoint_dir,
                                                'model.ckpt-' + str(hparams.checkpoint_number) + '.meta')

        # Restore tensorflow model saved in checkpoint file
        print("Restoring saved model")
        saver.restore(sess, checkpoint_filename)

    # Create checkpoint file to save models and variable values to
    TIMESTAMP = str(time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
    cur_model_dir = os.path.abspath(os.path.join(hparams.model_dir, str(TIMESTAMP)))
    checkpoint_filename = os.path.join(cur_model_dir, 'model.ckpt')

    # Create summary writer to save certain tensor values during training
    writer = tf.summary.FileWriter(cur_model_dir, sess.graph)

    # Start training loop with each iteration updating the network weights based on one input batch
    dice_score_list_background = []
    dice_score_list_bloodpool = []
    dice_score_list_wall = []
    dice_score_list_plaque = []
    dice_score_list_plaque_noise_filtered = []
    plaque_noise_counter = 0

    for step in range(3000000):
        print("The Batch Size is:  ",hparams.batch_size)
        print("Division is:  ", int(len(images_aug)/hparams.batch_size))

        if (hparams.augment_flag is True) and (step % int(len(images_aug)/hparams.batch_size) is 0) and (step >1) and (np.random.rand(1)[0] < 0.75):# all training before Feb 13 was at 0.75
            print('re-augment')
            images_aug, labels_aug = augment_n_reformat(images, labels, hparams)
            sess.run(input_images.initializer, feed_dict={x: images_aug})
            sess.run(input_labels.initializer, feed_dict={y: labels_aug})

        print('Step', step)
        start = time.time()
        # Train/update network weights based on current batch
        img_batch, gt_batch, prediction, _, summary, loss, dice_score_per_class, true_dice_score_per_class = sess.run([images_batch, labels_batch, logits, train_op, summary_op, total_loss, dice_coef_per_class, true_dice_coef_per_class])
        print("Dice score: " + str(true_dice_score_per_class))
        print("Loss: ", loss)

        # Remember Dice scores of previous steps to get average values
        dice_score_list_background.append(true_dice_score_per_class[0])
        dice_score_list_bloodpool.append(true_dice_score_per_class[1])
        dice_score_list_wall.append(true_dice_score_per_class[2])
        dice_score_list_plaque.append(true_dice_score_per_class[3])

        # Filter out noise cases from Dice calculation
        number_of_plaque_voxels_label = len(np.where(np.argmax(gt_batch[0, :, :, :], axis=2) == 3)[0])
        number_of_plaque_voxels_pred = len(np.where(np.argmax(prediction[0, :, :, :], axis=2) == 3)[0])

        number_of_myo_voxels_label = len(np.where(np.argmax(gt_batch[0, :, :, :], axis=2) == 2)[0])
        if number_of_plaque_voxels_label > 0.01 * number_of_myo_voxels_label:
            dice_score_list_plaque_noise_filtered.append(true_dice_score_per_class[3])

        print("Dice score: " + str(true_dice_score_per_class))
        print("Loss: "+ str(loss))

        '''
        print("Logits shape: " + str(logits_reshaped_2.shape))
        #print("Logits: " + str(logits_reshaped_2))
        print("Logits: " + str(logits_reshaped_2[0]))
        print("Weighted logits shape: " + str(weighted_logits_loss.shape))
        #print("Weighted logits: " + str(weighted_logits_loss))
        print("Weighted logits: " + str(weighted_logits_loss[0]))
        print("Argmax weighted : ")
        print(np.argmax(weighted_logits_loss[0]))
        print("pred:")
        print(prediction[0, 0, 0, 0, :])
        print("Argmax: ")
        print(np.argmax(prediction[0, 0, 0, 0, :]))
        print("class predicted in dice calculation: ")
        print(class_of_max_prediction[0])
        print("Ground truth value: ")
        print(gt_batch[0, 0, 0, 0, :])
        print("Argmax of grount truth: ")
        print(np.argmax(gt_batch[0, 0, 0, 0, :]))
        #print("Weighted loss: " + str(dice_coeffs_class_weighted_loss_all))
        '''

        print("Average dice score: " + str(np.average(dice_score_list_background)) + ", " + str(
            np.average(dice_score_list_bloodpool)) + ", " + str(np.average(dice_score_list_wall)) + ", " + str(
            np.average(dice_score_list_plaque)) + ", " + str(
            np.average(dice_score_list_plaque_noise_filtered)))

        end = time.time()
        dt = (end - start)
        print("Time to run one train step: ", dt)
        print(plaque_noise_counter)
        print(img_batch.shape)
        print(gt_batch.shape)
        print(prediction.shape)

        # Create images every 10 steps for easy checking
        if step % hparams.save_images_during_training_iteration_interval == 0:

            # Iterate over slices in current volume
            slice_predictions = []
            for slice_nr in range(img_batch.shape[3]):
                print("Evaluating slice number " + str(slice_nr))

                # Reformat batch image for later usage
                print(img_batch.shape)
                batch_img = img_batch[0, :, :, slice_nr, 0]
                print("batch image shape: ")
                print(batch_img.shape)

                # Get current slice predictions of network
                print("Handling slice with number " + str(slice_nr))
                print("Batch label type: " + str(type(gt_batch)))
                print("Batch label shape: " + str(gt_batch.shape))
                print("Batch sigmoid pred type: " + str(type(prediction)))
                print("Batch sigmoid pred shape: " + str(prediction.shape))
                print("Batch sigmoid shape of current slice of first volume in batch: " + str(prediction[0, :, :, slice_nr, :].shape))
                merged_label = np.argmax(gt_batch[0, :, :, slice_nr, :], axis=2)
                merged_pred = np.argmax(prediction[0, :, :, slice_nr, :], axis=2)
                print("Merged label type: " + str(type(merged_label)))
                print("Merged label shape: " + str(merged_label.shape))
                print("Merged label: " + str(merged_label))
                print("Merged pred type: " + str(type(merged_pred)))
                print("Merged pred shape: " + str(merged_pred.shape))
                print("Merged pred: " + str(merged_pred))
                print("Amount of pixels for each label class in merged label")
                print(len(np.where(merged_label == 0)[0]))
                print(len(np.where(merged_label == 1)[0]))
                print(len(np.where(merged_label == 2)[0]))
                print(len(np.where(merged_label == 3)[0]))
                print(len(np.where(merged_label == 4)[0]))
                print(len(np.where(merged_label >= 0)[0]))
                print("Amount of pixels for each label class in merged prediction")
                print(len(np.where(merged_pred == 0)[0]))
                print(len(np.where(merged_pred == 1)[0]))
                print(len(np.where(merged_pred == 2)[0]))
                print(len(np.where(merged_pred == 3)[0]))
                print(len(np.where(merged_pred == 4)[0]))
                print(len(np.where(merged_pred >= 0)[0]))
                slice_predictions.append(prediction[0, :, :, :, :])

                # Save current slice of first volume in batch
                if slice_nr == 0: # only one slice
                    feature_path = os.path.join(images_output_path, '_step_' + str(step) + '_' + str(slice_nr) + '_feature.jpg')
                    target_path = os.path.join(images_output_path, '_step_' + str(step) + '_' + str(slice_nr) + '_target.jpg')
                    pred_path = os.path.join(images_output_path, '_step_' + str(step) + '_' + str(slice_nr) + '_pred.jpg')
                    scipy.misc.imsave(feature_path, batch_img)
                    scipy.misc.toimage(merged_label, cmin=0.0, cmax=4).save(target_path)
                    scipy.misc.toimage(merged_pred, cmin=0.0, cmax=4).save(pred_path)
                    print("Saved slice images of feature, target and prediction of first volume in current batch")

        # Save model and create summaries
        if step % hparams.save_model_iteration_interval == 0:
            saver.save(sess, checkpoint_filename, global_step=global_step)
            writer.add_summary(summary, step)

        print('DONE WITH STEP')

    # Stop data input threads after finishing training step
    coord.request_stop()
    coord.join(threads)

