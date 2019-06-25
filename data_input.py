import os
import nibabel as nib
from collections import OrderedDict

def load_data_fnames(data_dir):
    label_identifier = 'label'
    # Get list of image files
    image_list = [f for f in sorted(os.listdir(data_dir)) if
                    os.path.isfile(os.path.join(data_dir, f)) and label_identifier not in f]
    # Get list of label files
    label_list = [f for f in sorted(os.listdir(data_dir)) if
                    os.path.isfile(os.path.join(data_dir, f)) and label_identifier in f]

    return image_list, label_list

def load_data(data_dir):
    images_list, labels_list = load_data_fnames(data_dir)
    # Read actual image
    images=[]
    labels=[]
    for i in range(len(images_list)):
        dd=nib.load(os.path.join(data_dir,images_list[i]))
        images.append(dd.get_fdata())
        dd=nib.load(os.path.join(data_dir,labels_list[i]))
        labels.append(dd.get_fdata())
    return images, labels

# Get images and labels and return them as dictionaries of the form (filename -> image) and (filename -> label)
def load_data_eval(data_dir):
    label_identifier = 'label'
    # Get list of image files
    image_list = [f for f in sorted(os.listdir(data_dir)) if
                    os.path.isfile(os.path.join(data_dir, f)) and label_identifier not in f]
    # Get list of label files
    label_list = [f for f in sorted(os.listdir(data_dir)) if
                    os.path.isfile(os.path.join(data_dir, f)) and label_identifier in f]

    # Read in all images from image file list
    images = OrderedDict()
    for f in image_list:
        img = nib.load(os.path.join(data_dir, f))
        images[f] = img

    # Read in all images from image file list
    labels = OrderedDict()
    for f in label_list:
        label = nib.load(os.path.join(data_dir, f))
        labels[f] = label

    return images, labels