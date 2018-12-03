import argparse
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import os


def visualize_semantic_segmentation(label_array, color_map, black_bg=False, save_path=None):
    """
    tool for visualizing semantic segmentation for a given label array

    :param label_array: [H, W], contains [0-nClasses], 0 for background
    :param color_map: array read from 'colorMapC46.mat'
    :param black_bg: the background is black if set True
    :param save_path: path for saving the image
    """
    visual_image = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    if not black_bg:
        visual_image.fill(255)

    ## read all colors
    colors_list = []
    for i in range(color_map.shape[0]):
        colors_list.append(color_map[i][1][0])
    colors_list = np.array(colors_list)

    ## assign color to drawing regions
    visual_image[label_array != 0] = colors_list[label_array[label_array != 0] - 1]

    plt.imshow(visual_image)
    plt.show()

    ## save visualization
    if save_path is not None:
        visual_image = Image.fromarray(visual_image, 'RGB')
        visual_image.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, choices=['train', 'val', 'test'],
                        default='train', help="choose a dataset")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image")
    parser.add_argument('--black_bg', '-bl', type=int, choices=[0, 1],
                        default=1, help="use black or white background for visualization")
    parser.add_argument('--data_basedir', '-db', type=str, default='../data', help="set the data base dir")

    args = parser.parse_args()

    if args.image_id == -1:
        raise Exception("An image should be chosen.")

    black_bg = True if args.black_bg == 1 else False

    ## load color map
    colorMap = scipy.io.loadmat(os.path.join(args.data_basedir, 'colorMapC46.mat'))['colorMap']

    ## load gt_label
    label_name = 'sample_' + str(args.image_id) + '_class.mat'  # e.g. sample_1_class.mat
    label_path = os.path.join(args.data_basedir, args.dataset, 'CLASS_GT', label_name)
    label = scipy.io.loadmat(label_path)['CLASS_GT']
    label = np.array(label, dtype=np.int32)  # shape = [H, W]

    visualize_save_base_dir = os.path.join(args.data_basedir, args.dataset, 'CLASS_GT_vis')
    os.makedirs(visualize_save_base_dir, exist_ok=True)
    visualize_semantic_segmentation(label, colorMap, black_bg=black_bg,
                                    save_path=os.path.join(visualize_save_base_dir, str(args.image_id) + '.png'))
