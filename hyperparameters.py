import tensorflow as tf
import numpy as np
from collections import namedtuple

# This file contains all hyperparameter settings used by the CNNs
#

# Input parameters
tf.flags.DEFINE_string("input_dir","./data/","Base directory for input data")

tf.flags.DEFINE_integer("volume_width", 128, "Width of volumes after resampling of input")
tf.flags.DEFINE_integer("volume_height", 128, "Height of volumes after resampling of input")
tf.flags.DEFINE_integer("volume_depth", 3, "Number of slices of the resampled volumes") # SlidingWindow =3
tf.flags.DEFINE_integer("num_labels", 4, "Number of different label classes in the ground truth of the input data")
tf.flags.DEFINE_integer("num_labels_output", 4, "Number of different label classes in the ground truth of the output data")

# Training parameters
tf.flags.DEFINE_boolean("augment_flag", True, "Defines whether to use augmentation or not") # ASF: add augmentation
tf.flags.DEFINE_integer("batch_size", 15, "Batch size during training")
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name")

# Model and image saving during training
tf.flags.DEFINE_integer("save_model_iteration_interval", 1000, "Number of training steps until a checkpoint of the current model shall be saved")
tf.flags.DEFINE_integer("save_images_during_training_iteration_interval", 1000, "Number of training steps until images or current batch shall be saved")

# Model restoration
tf.flags.DEFINE_boolean("restore_model", True, "Defines whether the model should be restored from a previous session")
tf.flags.DEFINE_string("model_dir", "./models/3D-LGE/", "Directory to save or restore trained models")
tf.flags.DEFINE_string("model_subdir", "20181224-201529", "Name of subdirectory of saved model to be loaded")
tf.flags.DEFINE_string("checkpoint_number", "107001", "Name of specific snapshot of saved model to be loaded")

# Loss
tf.flags.DEFINE_string("loss_type", "cross_entropy", "Loss type can be 'cross_entropy', 'dice' or 'jaccard'")
tf.flags.DEFINE_boolean("use_class_weighting", False, "Defines whether class weighting should be applied when calculating loss")
tf.flags.DEFINE_boolean("balance_scar_size",True, "weight the loss with scar size")

# Test parameters
tf.flags.DEFINE_string("results_dir", "./results/3D-LGE/WeightedLoss", "Directory to save segmentation outputs")
tf.flags.DEFINE_boolean("create_test_images", True, "Save image files of gt, pred, slice to results dir")
tf.flags.DEFINE_boolean("create_test_volumes", True, "Save volume files of gt, pred, slice to results dir")
tf.flags.DEFINE_boolean("save_prob_maps", True, "Save binary images or the raw probability maps") # setting to True for advanced processing

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
	"volume_size",
	"num_labels",
	"num_labels_output",
	"input_dir",
	"learning_rate",
	"batch_size",
	"optimizer",
	"model_dir",
	"restore_model",
	"augment_flag",
	"balance_scar_size",
	"model_subdir",
	"checkpoint_number",
    "save_model_iteration_interval",
	"save_images_during_training_iteration_interval",
	"loss_type",
	"use_class_weighting",
	"results_dir",
    "create_test_images",
    "save_prob_maps",
	"create_test_volumes",
    ])

def create_hparams():
  return HParams(
volume_size =              np.asarray([FLAGS.volume_width, FLAGS.volume_height, FLAGS.volume_depth]),
num_labels =               FLAGS.num_labels,
num_labels_output =        FLAGS.num_labels_output,
input_dir =                 FLAGS.input_dir,
learning_rate =             FLAGS.learning_rate,
batch_size =                FLAGS.batch_size,
optimizer =                 FLAGS.optimizer,
model_dir =                 FLAGS.model_dir,
restore_model =             FLAGS.restore_model,
augment_flag =				FLAGS.augment_flag,
balance_scar_size = 		FLAGS.balance_scar_size,
model_subdir =              FLAGS.model_subdir,
checkpoint_number =         FLAGS.checkpoint_number,
save_model_iteration_interval = FLAGS.save_model_iteration_interval,
save_images_during_training_iteration_interval = FLAGS.save_images_during_training_iteration_interval,
loss_type =                 FLAGS.loss_type,
use_class_weighting=		FLAGS.use_class_weighting,
results_dir =               FLAGS.results_dir,
create_test_images =        FLAGS.create_test_images,
save_prob_maps =            FLAGS.save_prob_maps,
create_test_volumes=FLAGS.create_test_volumes)
