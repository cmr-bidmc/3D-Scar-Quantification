from collections import OrderedDict
from data_input import load_data_eval
import re
import tensorflow as tf
import os
import numpy as np
import time
import scipy
import nibabel as nib
import hyperparameters
from model_3D import deep_model_128in as model

from data_reformating_3D import reformat_eval_data_3D as reformat_data_3D

from loss import loss


# Get hyperparameters
hparams = hyperparameters.create_hparams()

# Set data input path
test_data_dir =  hparams.input_dir + '/test'

# Load preprocessed input data
images_dict, labels_dict = load_data_eval(test_data_dir)


# Extract images from dict
images = []
for k, v in images_dict.items():
    images.append(v.get_data())

# Extract labels from dict
labels = []
for k, v in labels_dict.items():
    labels.append(v.get_data())

# Reformat input data
# Set up input data placeholder variables
x = tf.placeholder(dtype=tf.float32, shape=[1, hparams.volume_size[0], hparams.volume_size[1], hparams.volume_size[2],  1])
y = tf.placeholder(dtype=tf.float32, shape=[1, hparams.volume_size[0], hparams.volume_size[1], hparams.volume_size[2], hparams.num_labels])

# Feed input data batch into neural network architecture
logits, softmax = model(x, hparams)

# Calculate loss of current network
total_loss, dice_coef_per_class, true_dice_coef_per_class = loss(logits, y, hparams)

# Get checkpoint numbers of models to be evaluated
checkpoint_dir = os.path.join(os.path.abspath((hparams.model_dir)), hparams.model_subdir)
checkpoint_numbers = []

# Check if one single model is given
if not hparams.checkpoint_number:
    meta_file_identifier = 'meta'

    # Get list of checkpoint files
    checkpoint_meta_files = [f for f in sorted(os.listdir(checkpoint_dir)) if
                    os.path.isfile(os.path.join(checkpoint_dir, f)) and meta_file_identifier in f]
    print(checkpoint_meta_files)

    # Extract checkpoint numbers in directory from respective file names
    for meta_file in checkpoint_meta_files:
        cur_checkpoint_number = re.findall(r'\d+', meta_file)
        checkpoint_numbers.append(cur_checkpoint_number[0])
else:
    checkpoint_numbers = [str(hparams.checkpoint_number)]

print(checkpoint_numbers)

# Dice scores for each checkpoint
checkpoint_dice_list = OrderedDict()

for checkpoint_number in checkpoint_numbers:

    print("Evaluating checkpoint number ... " + str(checkpoint_number))
    checkpoint_filename = os.path.join(checkpoint_dir, 'model.ckpt-' + checkpoint_number)
    meta_checkpoint_filename = os.path.join(checkpoint_dir, 'model.ckpt-' + checkpoint_number + '.meta')


    # Evaluate the network
    with tf.Session() as sess:

        # Restore tensorflow model saved in checkpoint file
        print("Restoring saved model from file " + str(checkpoint_filename))
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_filename)

        # Setup variables to be used for later storage of metric values
        volume_counter = 0
        slice_dice_scores = {}
        total_inf_time = 0
        dice_score_list_background = []
        dice_score_list_bloodpool = []
        dice_score_list_wall = []
        dice_score_list_plaque = []
        dice_score_list_plaque_noise_filtered = []
        plaque_noise_counter = 0

        # Create helper variable volume_nr to find label corresponding to current volume
        volume_nr = 0

        # Iterate over all volumes in test data set
        for k, v in images_dict.items():

            print("Evaluating volume ...")
            print(k)

            # Declare two dicts to store volumes and images
            saved_volumes_dict = dict()
            saved_images_dict = dict()

            # Get current volume from dict
            full_feature_volume = images_dict[k].get_data()
            cur_case_name = k[0:-7]
            cur_label_name = cur_case_name + "_label.nii.gz"
            print("Getting label with name ... " + str(cur_label_name))
            print(full_feature_volume.shape)
            full_label_volume = labels_dict[cur_label_name].get_data()
            cur_nifti_feature_affine = images_dict[k].affine
            cur_nifti_label_affine = labels_dict[cur_label_name].affine

            # Get case number of current volume
            case_number = k.split('_')[0]

            # Declare variables
            slice_merged_lv_dice_scores = []
            slice_scar_dice_scores = []
            slice_scar_quantities_gt = []
            slice_scar_quantities_pred = []
            slice_merged_lv_quantities_gt = []
            slice_merged_lv_quantities_pred = []
            volume_counter += 1
            slice_counter = 0
            num_slices_in_volume = len(v.get_data())
            dt_vol = 0

            full_feature_volume, full_label_volume = reformat_data_3D(full_feature_volume, full_label_volume, hparams)

            # Initialize mid leftventricular start index
            mid_lv_start_index = num_slices_in_volume / 2 - 1
            if num_slices_in_volume > 10:
                mid_lv_start_index += 1

            # Create feed dict for tensorflow session input
            print("Defining feed dict")
            feed = {x: full_feature_volume, y: full_label_volume}

            # Get prediction of current network
            start = time.time()
            print("Get network prediction for current slice")
            loss, dice_score_per_class, true_dice_score_per_class, batch_image, batch_label, batch_sigmoid_pred = sess.run(
                    [total_loss, dice_coef_per_class, true_dice_coef_per_class, x, y, softmax], feed_dict=feed)
            print("Dice score: " + str(dice_score_per_class))
            print("Loss: ", loss)
            end = time.time()
            dt = (end - start)
            dt_vol += dt
            print("Time to run one train step with one input slice: ", dt)

            # Save Dice scores of current volume in array
            dice_score_list_background.append(true_dice_score_per_class[0])
            dice_score_list_bloodpool.append(true_dice_score_per_class[1])
            dice_score_list_wall.append(true_dice_score_per_class[2])
            dice_score_list_plaque.append(true_dice_score_per_class[3])

            # Filter out noise cases from Dice calculation
            print("Batch label shape: " + str(batch_label.shape))
            print("Batch sigmoid prediction: " + str(batch_sigmoid_pred))
            number_of_plaque_voxels_label = len(np.where(np.argmax(batch_label[0, :, :, :], axis=2) == 3)[0])
            number_of_plaque_voxels_pred = len(np.where(np.argmax(batch_sigmoid_pred[0, :, :, :], axis=2) == 3)[0])

            number_of_myo_voxels_label = len(np.where(np.argmax(batch_label[0, :, :, :], axis=2) == 2)[0])
            if number_of_plaque_voxels_label > 0.01* number_of_myo_voxels_label:
                dice_score_list_plaque_noise_filtered.append(true_dice_score_per_class[3])


            print("Average dice score: " + str(np.average(dice_score_list_background)) + ", " + str(
                np.average(dice_score_list_bloodpool)) + ", " + str(np.average(dice_score_list_wall)) + ", " + str(
                np.average(dice_score_list_plaque)) + ", " + str(
                np.average(dice_score_list_plaque_noise_filtered)))
            print(plaque_noise_counter)

            # Iterate over slices in current volume
            slice_predictions = []
            for slice_nr in range(batch_image.shape[3]):
                print("Evaluating slice number " + str(slice_nr))

                # Reformat batch image for later usage
                print(batch_image.shape)
                batch_img = batch_image[0, :, :, slice_nr, 0]
                print("batch image shape: ")
                print(batch_img.shape)

                # Get current slice predictions of network
                print("Handling slice with number " + str(slice_nr) + " from " + str(num_slices_in_volume))
                print("Batch label type: " + str(type(batch_label)))
                print("Batch label shape: " + str(batch_label.shape))
                print("Batch sigmoid pred type: " + str(type(batch_sigmoid_pred)))
                print("Batch sigmoid pred shape: " + str(batch_sigmoid_pred.shape))
                merged_label = np.argmax(batch_label[0, :, :, slice_nr, :], axis=2)
                merged_pred = np.argmax(batch_sigmoid_pred[0, :, :, slice_nr, :], axis=2)
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
                slice_predictions.append(batch_sigmoid_pred[0, ...])

                # Save current slice feature, target and prediction as jpeg files if respective hyperparameter is set to true
                if hparams.create_test_images == True:

                    # Get directory and respective file names
                    current_results_dir_base = os.path.join(hparams.results_dir, hparams.model_subdir)
                    current_results_dir_iteration = os.path.join(current_results_dir_base, 'checkpoint_data',
                                                                 'images_' + str(hparams.volume_size[0]))
                    current_results_dir_test_data = os.path.join(current_results_dir_base, 'test_data',
                                                                 'images_' + str(hparams.volume_size[0]))
                    if not os.path.exists(current_results_dir_test_data):
                        os.makedirs(current_results_dir_test_data)
                    if not os.path.exists(current_results_dir_iteration):
                        os.makedirs(current_results_dir_iteration)
                    feature_path = os.path.join(current_results_dir_test_data,
                                                case_number + '_slice' + str(slice_nr) + '_feature.jpg')
                    target_path = os.path.join(current_results_dir_test_data,
                                               case_number + '_slice' + str(slice_nr) + '_target.jpg')
                    pred_path = os.path.join(current_results_dir_iteration,
                                             case_number + '_slice' + str(slice_nr) + '_pred.jpg')

                    # Perform actual saving operation
                    scipy.misc.imsave(feature_path, batch_img)
                    scipy.misc.toimage(merged_label, cmin=0.0, cmax=4).save(target_path)
                    scipy.misc.toimage(merged_pred, cmin=0.0, cmax=4).save(pred_path)
                    saved_images_dict[case_number] = {'feature': feature_path, 'target': target_path, 'pred': pred_path}

                slice_counter += 1

            # Add process time for current volume to total time
            total_inf_time += dt_vol

            # Create volume prediction out of individual slice predictions
            print("Slice predictions shape: " + str((np.asarray(slice_predictions)).shape))
            if hparams.save_prob_maps is False:
                merged_volume_prediction = np.argmax(batch_sigmoid_pred[0, :, :, :, :], axis=3)
            else:
                merged_volume_prediction = batch_sigmoid_pred[0, :, :, :, :]

            print(merged_volume_prediction.shape)
            print(full_label_volume.shape)
            print(full_feature_volume.shape)
            print(type(merged_volume_prediction))
            print(type(full_label_volume))
            print(type(full_feature_volume))

            # Save current feature, target and prediction volume as nifti files if respective hyperparameter is set to true
            if hparams.create_test_volumes is True:
                current_results_dir_base = os.path.join(hparams.results_dir, hparams.model_subdir)
                current_results_dir_nifti_features = os.path.join(current_results_dir_base, k + str("_feature.nii.gz"))
                current_results_dir_nifti_labels = os.path.join(current_results_dir_base, k + str("_label.nii.gz"))
                current_results_dir_nifti_pred = os.path.join(current_results_dir_base, k + str("_pred.nii.gz"))
                nib.save(nib.Nifti1Image(merged_volume_prediction, cur_nifti_label_affine), current_results_dir_nifti_pred)
                nib.save(nib.Nifti1Image(full_feature_volume, cur_nifti_feature_affine), current_results_dir_nifti_features)
                nib.save(nib.Nifti1Image(full_label_volume, cur_nifti_label_affine), current_results_dir_nifti_labels)

    # Print Dice scores of current checkpoint
    checkpoint_dice_list[checkpoint_number] = [np.average(dice_score_list_background), np.average(dice_score_list_bloodpool), np.average(dice_score_list_wall), np.average(dice_score_list_plaque), np.average(dice_score_list_plaque_noise_filtered)]
    print("Dice values at current checkpoint: ")
    print(checkpoint_dice_list)
