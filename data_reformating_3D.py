from skimage import transform
import numpy as np

def reformat_eval_data_3D(images, labels, hparams):
    lbl64 = labels
    img64 = images
    L = labels.shape
    lx = L[0]
    ly = L[1]
    lbl_c = lbl64[lx // 4: - lx // 4, ly // 4: - ly // 4, :]
    lbl64 = transform.resize(lbl_c, (hparams.volume_size[0], hparams.volume_size[1], 3), order=0)

    img_c = img64[lx // 4: - lx // 4, ly // 4: - ly // 4, :]
    img64 = transform.resize(img_c, (hparams.volume_size[0], hparams.volume_size[1], 3), mode='reflect')

    print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
    print(images.shape)
    print(labels.shape)

    images=img64
    labels=lbl64

    feature_slices = np.stack(images)
    label_slices = np.stack(labels)

    print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
    print(feature_slices.shape)
    print(label_slices.shape)
    reshaped_feature_volumes = np.zeros(
        [1, feature_slices.shape[0], feature_slices.shape[1], feature_slices.shape[2], 1])
    reshaped_feature_volumes[0,:, :, :, 0] = feature_slices
    reshaped_label_volumes = np.zeros([label_slices.shape[0], label_slices.shape[1], label_slices.shape[2]])
    reshaped_label_volumes[:, :, :] = label_slices
    print(reshaped_feature_volumes.shape)
    print(reshaped_label_volumes.shape)

    # split label into separate classes(labels_batch)
    label_merged = reshaped_label_volumes
    split_labels = np.zeros([1,label_merged.shape[0], label_merged.shape[1], label_merged.shape[2], hparams.num_labels_output])
    for label_number in range(hparams.num_labels_output):
        class_pixels = (label_merged == label_number)
        split_labels[0,..., label_number][class_pixels] = 1

    print("Split labels batch output")
    print(split_labels.shape)

    # Perform other preprocessing here
    print(reshaped_feature_volumes.shape)
    print(len(reshaped_feature_volumes))
    print(reshaped_label_volumes.shape)
    print(len(reshaped_label_volumes))

    return reshaped_feature_volumes, split_labels