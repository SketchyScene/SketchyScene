import os
import argparse
import sys
sys.path.append('..')
sys.path.append('../libs')
import utils
import visualize
from model import log
from SketchDataset import SketchDataset


def visualize_instance_segmentation(data_base_dir, dataset_type, image_id, save_path='', verbose=True):
    split_dataset = SketchDataset(data_base_dir)
    split_dataset.load_sketches(dataset_type)
    split_dataset.prepare()

    original_image = split_dataset.load_image(image_id - 1)
    gt_mask, gt_class_id = split_dataset.load_mask(image_id - 1)
    gt_bbox = utils.extract_bboxes(gt_mask)

    if verbose:
        log('original_image', original_image)
        log('gt_class_id', gt_class_id)
        log('gt_bbox', gt_bbox)
        log('gt_mask', gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                split_dataset.class_names, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, choices=['train', 'val', 'test'],
                        default='train', help="choose a dataset")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image")
    parser.add_argument('--data_basedir', '-db', type=str, default='../../data', help="set the data base dir")

    args = parser.parse_args()

    if args.image_id == -1:
        raise Exception("An image should be chosen.")

    visualize_save_base_dir = os.path.join(args.data_basedir, args.dataset, 'INSTANCE_GT_vis')
    os.makedirs(visualize_save_base_dir, exist_ok=True)

    visualize_instance_segmentation(args.data_basedir, args.dataset, args.image_id,
                                    save_path=os.path.join(visualize_save_base_dir, str(args.image_id) + '.png'))