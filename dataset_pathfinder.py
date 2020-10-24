# ---------------------------------------------------------------------------------------
# Pytorch Dataset for an already generated Pathfinder on Natural Images Dataset.
# This is the one to use in training models.
#
# For the 'online' pathfinder on natural Images Dataset, see
# generate_pathfinder_dataset.py. Currently this is only used for data generation.
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transform_functional
from torch.utils.data import DataLoader, Dataset
import utils


class PathfinderNaturalImages(Dataset):

    def __init__(self, data_dir, transform=None, subset_size=None):

        self.transform = transform

        imgs_dir = os.path.join(data_dir, 'images')
        labels_file = os.path.join(data_dir, 'classification_labels.txt')
        data_key_file = os.path.join(data_dir, 'data_key.txt')
        required_items = [imgs_dir, labels_file, data_key_file]
        for item in required_items:
            if not os.path.exists(item):
                raise Exception('Required item {} not found'.format(item))

        debug_indv_contours_labels_dir = os.path.join(data_dir, 'individual_contours_labels')
        debug_full_labels_dir = os.path.join(data_dir, 'full_labels')
        debug_extra_info_dir = os.path.join(data_dir, 'extra_info')
        debug_items = [debug_indv_contours_labels_dir, debug_full_labels_dir, debug_extra_info_dir]
        for item in debug_items:
            if not os.path.exists(item):
                raise Exception('Debug item {} not found'.format(item))

        with open(data_key_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(imgs_dir, file) for file in file_names]
        self.indv_contours_labels = \
            [os.path.join(debug_indv_contours_labels_dir, file) for file in file_names]
        self.full_labels = [os.path.join(debug_full_labels_dir, file) for file in file_names]
        self.extra_info_files = \
            [os.path.join(debug_extra_info_dir, file.split('.')[0] + '.pickle')
             for file in file_names]

        with open(labels_file, "r") as f:
            self.labels = [int(x.strip()) for x in f.readlines()]

        if subset_size is not None:
            assert subset_size <= len(
                self.images), 'subset size {} is greater than dataset size {}'.format(
                subset_size, len(self.images))

            use_idxs = np.random.choice(
                np.arange(len(self.images)), size=subset_size, replace=False)

            self.images = [self.images[idx] for idx in use_idxs]
            self.indv_contours_labels = [self.indv_contours_labels[idx] for idx in use_idxs]
            self.full_labels = [self.full_labels[idx] for idx in use_idxs]
            self.labels = [self.labels[idx] for idx in use_idxs]
            self.extra_info_files = [self.extra_info_files[idx] for idx in use_idxs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        img = transform_functional.to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        label = torch.tensor(label)

        indv_contours_label = Image.open(self.indv_contours_labels[index]).convert('L')
        indv_contours_label = transform_functional.to_tensor(indv_contours_label)

        f_label = Image.open(self.full_labels[index]).convert('L')
        f_label = transform_functional.to_tensor(f_label)

        with open(self.extra_info_files[index], 'rb') as handle:
            extra_info = pickle.load(handle)

        d = extra_info['ep_distance']
        d = torch.tensor(d)

        return img, label, indv_contours_label, f_label, d


if __name__ == "__main__":
    data_set_dir = './data/pathfinder_natural_images_test'
    random_seed = 5

    save_dir = './results/sample_images'

    # Immutable ----------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # -----------------------------------------------------------------------------------
    # Data loader
    # -----------------------------------------------------------------------------------
    ch_mean = [0.485, 0.456, 0.406]  # Imagenet Values
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    transforms_list = [
        transforms.Normalize(mean=ch_mean, std=ch_std),
        utils.PunctureImage(n_bubbles=200, fwhm=np.array([7, 9, 11, 13, 15, 17]))
    ]
    pre_process_transforms = transforms.Compose(transforms_list)

    dataset = PathfinderNaturalImages(
        os.path.join(data_set_dir, 'train'),
        transform=pre_process_transforms,
        subset_size=50
    )

    data_loader = DataLoader(
        dataset=dataset,
        num_workers=0,
        batch_size=32,
        shuffle=False,  # Do not change for correct image mapping
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    index1 = 0
    for iteration, data_loader_out in enumerate(data_loader, 0):

        imgs, labels, individual_contours_labels, full_labels, distances = data_loader_out

        image = imgs[index1, ]
        class_label = labels[index1, ]
        single_contour_label = individual_contours_labels[index1, ]
        full_label = full_labels[index1, ]
        distance = distances[index1, ]

        image = np.transpose(image, axes=(1, 2, 0))
        display_img = (image - image.min()) / (image.max() - image.min())

        fig, ax_arr = plt.subplots(1, 3, figsize=(15, 5))

        ax_arr[0].imshow(display_img)
        ax_arr[1].imshow(np.squeeze(single_contour_label))
        ax_arr[1].set_title("Connected {}.".format(bool(class_label)))
        ax_arr[2].imshow(np.squeeze(full_label))

        fig.suptitle("Distance between points {}\n".format(distance))
        fig.savefig(os.path.join(save_dir, 'img_{}.png'.format(iteration)))

        import pdb
        pdb.set_trace()

        plt.close(fig)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print('End')
    import pdb
    pdb.set_trace()
