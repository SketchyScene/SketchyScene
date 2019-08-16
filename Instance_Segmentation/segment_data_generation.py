import os
import sys
import json
import scipy.io
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse

sys.path.append('libs')

from config import Config
import model as modellib
from model import log
import visualize
from SketchDataset import SketchDataset
from edgelist_utils import refine_mask_with_edgelist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))


class SkeSegConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sketchyscene"

    # Train on 1 GPU and 16 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 16 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 46  # background + 46 classes

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False

    # image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

    # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # use the binary input to filter the pred_mask if 'True'
    IGNORE_BG = True


def segment_data_generation(mode, data_base_dir, use_edgelist=False, debug=False):
    if mode == 'both':
        dataset_types = ['val', 'test']
    else:
        dataset_types = [mode]

    caption_base_dir = 'data'
    outputs_base_dir = 'outputs'
    trained_model_dir = os.path.join(outputs_base_dir, 'snapshot')
    edgelist_result_dir = os.path.join(outputs_base_dir, 'edgelist')
    seg_data_save_base_dir = os.path.join(outputs_base_dir, 'inst_segm_output_data')
    epochs = '0100'
    model_path = os.path.join(trained_model_dir, 'mask_rcnn_sketchyscene_' + epochs + '.h5')

    dataset_class_names = ['bg']
    color_map_mat_path = os.path.join(data_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(46):
        cat_name = colorMap[i][0][0]
        dataset_class_names.append(cat_name)

    ROAD_LABEL = dataset_class_names.index('road')

    CLASS_ORDERS = [[dataset_class_names.index('sun'), dataset_class_names.index('moon'),
                     dataset_class_names.index('star'), dataset_class_names.index('road')],
                    [dataset_class_names.index('tree')],
                    [dataset_class_names.index('cloud')],
                    [dataset_class_names.index('house')],
                    [dataset_class_names.index('bus'), dataset_class_names.index('car'),
                     dataset_class_names.index('truck')]]

    config = SkeSegConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='', log_dir='')

    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    for dataset_type in dataset_types:
        caption_json_path = os.path.join(caption_base_dir, 'sentence_instance_' + dataset_type + '.json')

        fp = open(caption_json_path, "r")
        json_data = fp.read()
        json_data = json.loads(json_data)
        print('data_len', len(json_data))

        # val/test dataset
        dataset = SketchDataset(data_base_dir)
        dataset.load_sketches(dataset_type)
        dataset.prepare()

        split_seg_data_save_base_dir = os.path.join(seg_data_save_base_dir, dataset_type)
        os.makedirs(split_seg_data_save_base_dir, exist_ok=True)

        for data_idx in range(len(json_data)):
            img_idx = json_data[data_idx]['key']
            print('Processing', dataset_type, data_idx + 1, '/', len(json_data))

            original_image, _, gt_class_id, gt_bbox, gt_mask, _ = \
                modellib.load_image_gt(dataset, config, img_idx - 1, use_mini_mask=False)

            ## 1. inference
            results = model.detect([original_image])
            r = results[0]

            pred_boxes = r["rois"]  # (nRoIs, (y1, x1, y2, x2))
            pred_class_ids = r["class_ids"]  # (nRoIs)
            pred_scores = r["scores"]
            pred_masks = r["masks"]  # (768, 768, nRoIs)

            log("pred_boxes", pred_boxes)
            log("pred_class_ids", pred_class_ids)
            log("pred_masks", pred_masks)

            ## 2. Use original_image(768, 768, 3) {0, 255} to filter pred_masks
            if config.IGNORE_BG:
                pred_masks = np.transpose(pred_masks, (2, 0, 1))  # (nRoIs, 768, 768)
                bin_input = original_image[:, :, 0] == 255
                pred_masks[:, bin_input[:, :]] = 0  # (nRoIs, 768, 768)
                pred_masks = np.transpose(pred_masks, (1, 2, 0))  # (768, 768, nRoIs)

            if debug:
                visualize.display_instances(original_image, pred_boxes, pred_masks, pred_class_ids,
                                            dataset.class_names, pred_scores, figsize=(8, 8))

            ## 3. refine pred_masks(768, 768, nRoIs) with edge-list
            if use_edgelist:
                pred_masks = \
                    refine_mask_with_edgelist(img_idx, dataset_type, data_base_dir, edgelist_result_dir,
                                              pred_masks.copy(), pred_boxes)

            ## 4. TODO: remove road prediction
            # pred_boxes = pred_boxes.tolist()
            # pred_masks = np.transpose(pred_masks, (2, 0, 1)).tolist()
            # pred_scores = pred_scores.tolist()
            # pred_class_ids = pred_class_ids.tolist()
            #
            # while ROAD_LABEL in pred_class_ids:
            #     road_idx = pred_class_ids.index(ROAD_LABEL)
            #     pred_boxes.remove(pred_boxes[road_idx])
            #     pred_masks.remove(pred_masks[road_idx])
            #     pred_scores.remove(pred_scores[road_idx])
            #     pred_class_ids.remove(ROAD_LABEL)

            ## 5. TODO: add road from semantic prediction
            # sem_label_base_path = '../../../../Sketch-Segmentation-TF/Segment-Sketch-DeepLab-v2/edge-list/pred_semantic_label_edgelist/'
            # sem_label_base_path = os.path.join(sem_label_base_path, dataset_type, 'mat')
            # sem_label_path = os.path.join(sem_label_base_path, 'L0_sample' + str(img_idx) + '.mat')
            # sem_label = scipy.io.loadmat(sem_label_path)['pred_label_edgelist']  # (750, 750), [0, 46]
            #
            # if ROAD_LABEL in sem_label:
            #     road_mask_img = np.zeros([sem_label.shape[0], sem_label.shape[1], 3], dtype=np.uint8)
            #     road_mask_img[sem_label == ROAD_LABEL] = [255, 255, 255]  # (750, 750, 3), {0, 255}
            #     road_mask_img = scipy.misc.imresize(
            #         road_mask_img, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), interp='nearest')  # (768, 768, 3)
            #     road_mask = np.zeros(road_mask_img[:, :, 0].shape, dtype=np.uint8)
            #     road_mask[road_mask_img[:, :, 0] == 255] = 1  # (768, 768), {0, 1}
            #     # plt.imshow(road_mask)
            #     # plt.show()
            #
            #     road_bbox = utils.extract_bboxes(np.expand_dims(road_mask, axis=2))  # [num_instances, (y1, x1, y2, x2)]
            #     road_bbox = road_bbox[0]
            #     pred_boxes.append(road_bbox)
            #     pred_masks.append(road_mask)
            #     pred_scores.append(1.)
            #     pred_class_ids.append(ROAD_LABEL)

            # pred_boxes = np.array(pred_boxes, dtype=np.int32)
            # pred_class_ids = np.array(pred_class_ids, dtype=np.int32)
            # pred_scores = np.array(pred_scores, dtype=np.float32)
            # pred_masks = np.array(pred_masks, dtype=np.uint8)
            # pred_masks = np.transpose(pred_masks, [1, 2, 0])  # (768, 768, nRoIs?)

            if debug:
                visualize.display_instances(original_image, pred_boxes, pred_masks, pred_class_ids,
                                            dataset.class_names, pred_scores, figsize=(8, 8))

            ## 8. sort instances
            instance_sorted_index = []

            for order_idx in range(len(CLASS_ORDERS)):
                order_ids = CLASS_ORDERS[order_idx]
                for cate_idx in range(pred_class_ids.shape[0]):
                    if pred_class_ids[cate_idx] in order_ids:
                        instance_sorted_index.append(cate_idx)

            for cate_idx in range(pred_class_ids.shape[0]):
                if cate_idx not in instance_sorted_index:
                    instance_sorted_index.append(cate_idx)

            # print('pred_class_ids', pred_class_ids)
            # print('instance_sorted_index', instance_sorted_index)
            assert len(instance_sorted_index) == pred_class_ids.shape[0]

            pred_class_ids_list = []
            pred_masks_list = []
            pred_boxes_list = []

            for cate_idx_i in range(len(instance_sorted_index)):
                pred_class_ids_list.append(pred_class_ids[instance_sorted_index[cate_idx_i]])
                pred_masks_list.append(pred_masks[:, :, instance_sorted_index[cate_idx_i]])
                pred_boxes_list.append(pred_boxes[instance_sorted_index[cate_idx_i]])

            # print('pred_class_ids_list', pred_class_ids_list)
            assert len(pred_class_ids_list) == pred_class_ids.shape[0]

            ## 9. generate .npz data
            npz_name = os.path.join(split_seg_data_save_base_dir, str(img_idx) + '_datas.npz')
            np.savez(npz_name, pred_class_ids=pred_class_ids_list, pred_masks=pred_masks_list,
                     pred_boxes=pred_boxes_list)


def debug_saved_npz(dataset_type, img_idx, data_base_dir):
    outputs_base_dir = 'outputs'
    seg_data_save_base_dir = os.path.join(outputs_base_dir, 'inst_segm_output_data', dataset_type)

    npz_name = os.path.join(seg_data_save_base_dir, str(img_idx) + '_datas.npz')
    npz = np.load(npz_name)

    pred_class_ids = np.array(npz['pred_class_ids'], dtype=np.int32)
    pred_masks = np.array(npz['pred_masks'], dtype=np.uint8)
    pred_boxes = np.array(npz['pred_boxes'], dtype=np.int32)
    pred_masks = np.transpose(pred_masks, (1, 2, 0))
    print(pred_class_ids.shape)
    print(pred_masks.shape)
    print(pred_boxes.shape)

    image_name = 'L0_sample' + str(img_idx) + '.png'
    images_base_dir = os.path.join(data_base_dir, dataset_type, 'DRAWING_GT')
    image_path = os.path.join(images_base_dir, image_name)
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((768, 768), resample=Image.NEAREST)
    original_image = np.array(original_image, dtype=np.float32)  # shape = [H, W, 3]

    dataset_class_names = ['bg']
    color_map_mat_path = os.path.join(data_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(46):
        cat_name = colorMap[i][0][0]
        dataset_class_names.append(cat_name)

    visualize.display_instances(original_image, pred_boxes, pred_masks, pred_class_ids,
                                dataset_class_names, figsize=(8, 8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_basedir', '-db', type=str, default='../data', help="set the data base dir")
    parser.add_argument('--dataset', '-ds', type=str, choices=['val', 'test', 'both'],
                        default='both', help="choose a dataset")
    parser.add_argument('--use_edgelist', '-el', type=int, choices=[0, 1],
                        default=1, help="use edgelist or not")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image for debug")
    args = parser.parse_args()

    segment_data_generation(mode=args.dataset,
                            data_base_dir=args.data_basedir,
                            use_edgelist=args.use_edgelist)

    ## For debugging
    # assert args.dataset in ['val', 'test']
    # assert args.image_id != -1
    # debug_saved_npz(dataset_type=args.dataset,
    #                 img_idx=args.image_id,
    #                 data_base_dir=args.data_basedir)
