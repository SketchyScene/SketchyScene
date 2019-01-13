import os
import sys
import numpy as np
import scipy.io
from PIL import Image

sys.path.append('libs')
import utils

nImgs_map = {'train': 5617, 'val': 535, 'test': 1113}


class SketchDataset(utils.Dataset):
    """Generates the sketchyscene dataset."""

    def __init__(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir
        color_map_mat_path = os.path.join(dataset_base_dir, 'colorMapC46.mat')
        self.colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
        super(SketchDataset, self).__init__()

    def load_sketches(self, mode):
        assert mode in ["train", "val", "test"]

        # Add classes
        for i in range(46):
            cat_name = self.colorMap[i][0][0]
            self.add_class("sketchyscene", i + 1, cat_name)

        # Add images
        nImgs = nImgs_map[mode]

        for i in range(nImgs):
            self.add_image("sketchyscene", image_id=i, path="", mode=mode)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        info = self.image_info[image_id]
        mode = info['mode']

        image_name = 'L0_sample' + str(image_id + 1) + '.png'  # e.g. L0_sample5564.png

        images_base_dir = os.path.join(self.dataset_base_dir, mode, 'DRAWING_GT')
        image_path = os.path.join(images_base_dir, image_name)
        # print(image_path)
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image, dtype=np.float32)  # shape = [H, W, 3]
        # plt.imshow(image.astype(np.uint8))  #
        # plt.show()
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sketchyscene":
            return info['mode']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mode = info['mode']

        mask_class_name = 'sample_' + str(image_id + 1) + '_class.mat'
        mask_instance_name = 'sample_' + str(image_id + 1) + '_instance.mat'

        class_base_dir = os.path.join(self.dataset_base_dir, mode, 'CLASS_GT')
        instance_base_dir = os.path.join(self.dataset_base_dir, mode, 'INSTANCE_GT')
        mask_class_path = os.path.join(class_base_dir, mask_class_name)
        mask_instance_path = os.path.join(instance_base_dir, mask_instance_name)

        INSTANCE_GT = scipy.io.loadmat(mask_instance_path)['INSTANCE_GT']
        INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
        CLASS_GT = scipy.io.loadmat(mask_class_path)['CLASS_GT']  # (750, 750)

        # print(np.max(INSTANCE_GT))  # e.g. 101
        instance_count = np.bincount(INSTANCE_GT.flatten())
        # print(instance_count.shape)  # e.g. shape=(102,)

        instance_count = instance_count[1:]  # e.g. shape=(101,)
        nonzero_count = np.count_nonzero(instance_count)  # e.g. 16
        # print("nonzero_count", nonzero_count)  # e.g. shape=(102,)

        mask_set = np.zeros([nonzero_count, INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
        class_id_set = np.zeros([nonzero_count], dtype=np.uint8)

        real_instanceIdx = 0
        for i in range(instance_count.shape[0]):
            if instance_count[i] == 0:
                continue

            instanceIdx = i + 1

            ## mask
            mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
            mask[INSTANCE_GT == instanceIdx] = 1
            mask_set[real_instanceIdx] = mask

            class_gt_filtered = CLASS_GT * mask
            class_gt_filtered = np.bincount(class_gt_filtered.flatten())
            class_gt_filtered = class_gt_filtered[1:]
            class_id = np.argmax(class_gt_filtered) + 1

            class_id_set[real_instanceIdx] = class_id

            real_instanceIdx += 1

        mask_set = np.transpose(mask_set, (1, 2, 0))

        return mask_set, class_id_set
