import os
import sys
import tensorflow as tf
import argparse
import time
import numpy as np
from datetime import timedelta

sys.path.append('libs')
from config import Config
import utils
import model as modellib
from edgelist_utils import refine_mask_with_edgelist
from SketchDataset import SketchDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))


class SketchEvalConfig(Config):
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
    USE_MINI_MASK = True

    # image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

    # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # use the binary input to filter the pred_mask if 'True'
    IGNORE_BG = True


def segment_evaluate(**kwargs):
    data_base_dir = kwargs['data_base_dir']
    dataset_type = kwargs['dataset_type']
    epochs = kwargs['epochs']
    use_edgelist = kwargs['use_edgelist']

    iou_threshold = None  # None for mAP@[0.5:0.95]

    outputs_base_dir = 'outputs'
    eval_result_save_dir = os.path.join(outputs_base_dir, 'eval_result')
    edgelist_result_dir = os.path.join(outputs_base_dir, 'edgelist')
    trained_model_dir = os.path.join(outputs_base_dir, 'snapshot')
    model_path = os.path.join(trained_model_dir, 'mask_rcnn_sketchyscene_' + epochs + '.h5')
    
    os.makedirs(eval_result_save_dir, exist_ok=True)

    config = SketchEvalConfig()
    config.display()

    # val/test dataset
    dataset_eval = SketchDataset(data_base_dir)
    dataset_eval.load_sketches(dataset_type)
    dataset_eval.prepare()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='', log_dir='')

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Compute VOC-Style mAP @ IoU=0.5
    nImages = dataset_eval.num_images
    APs = []
    APs_edg = []
    time_sec_used = 0
    for i in range(nImages):
        start_time = time.time()
        image_id = i + 1
        print('Processing', image_id, '/', nImages)
        original_image, _, gt_class_id, gt_bbox, gt_mask, _ = \
            modellib.load_image_gt(dataset_eval, config, i, use_mini_mask=False)

        # Run object detection
        results = model.detect([original_image], verbose=0)
        r = results[0]
        pred_boxes = r["rois"]  # (nRoIs, (y1, x1, y2, x2))
        pred_class_ids = r["class_ids"]  # (nRoIs)
        pred_scores = r["scores"]  # (nRoIs)
        pred_masks = r["masks"]  # (768, 768, nRoIs)

        if config.IGNORE_BG:
            # Use original_image(768, 768, 3) {0, 255} to filter pred_masks
            pred_masks = np.transpose(pred_masks, (2, 0, 1))  # (nRoIs, 768, 768)
            bin_input = original_image[:, :, 0] == 255
            pred_masks[:, bin_input[:, :]] = 0  # (nRoIs, 768, 768)
            pred_masks = np.transpose(pred_masks, (1, 2, 0))  # (768, 768, nRoIs)

        # refine pred_masks(768, 768, nRoIs) with edge-list
        if use_edgelist:
            refined_pred_masks = \
                refine_mask_with_edgelist(image_id, dataset_type, data_base_dir, edgelist_result_dir,
                                        pred_masks.copy(), pred_boxes)

        # Compute AP
        if iou_threshold is None:
            iou_thresholds = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
            AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            AP_list_edg = np.zeros([len(iou_thresholds)], dtype=np.float32)
            for j in range(len(iou_thresholds)):
                iouThr = iou_thresholds[j]
                AP_single_iouThr, precisions, recalls, overlaps = \
                    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                     pred_boxes, pred_class_ids, pred_scores, pred_masks,
                                     iou_threshold=iouThr)
                AP_list[j] = AP_single_iouThr

                if use_edgelist:
                    AP_single_iouThr_edg, precisions, recalls, overlaps = \
                        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                         pred_boxes, pred_class_ids, pred_scores, refined_pred_masks,
                                         iou_threshold=iouThr)
                    AP_list_edg[j] = AP_single_iouThr_edg

            AP = AP_list
            AP_edg = AP_list_edg
            print('mAP', np.mean(AP))
            print('mAP_edg', np.mean(AP_edg))
        else:
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 pred_boxes, pred_class_ids, pred_scores, pred_masks,
                                 iou_threshold=iou_threshold)

            if use_edgelist:
                AP_edg, precisions, recalls, overlaps = \
                    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                     pred_boxes, pred_class_ids, pred_scores, refined_pred_masks,
                                     iou_threshold=iou_threshold)
            else:
                AP_edg = None

        APs.append(AP)
        APs_edg.append(AP_edg)

        # count left time
        end_time = time.time()
        time_sec_used += end_time - start_time
        rate = time_sec_used / image_id
        left_sec = (nImages - image_id) * rate
        print("single image times used: %f, left time: %s"
              % (end_time - start_time, str(timedelta(seconds=left_sec))))

    mAP = np.mean(APs)
    mAP_list = np.mean(APs, axis=0)
    if iou_threshold is None:
        iou_str = '@[0.5:0.95]'
    else:
        iou_str = '@[' + str(iou_threshold) + ']'

    print(epochs, "epochs model, ", "iou_threshold: ", iou_str, ' #original_mask')
    print("mAP: ", mAP)
    outstr = '## ' + dataset_type + ' datas\n'
    outstr += epochs + ' epochs model,' + ' iou_threshold: ' + iou_str + ' #original_mask' + '\n'
    outstr += 'mAP: ' + str(mAP) + '\n'

    if iou_threshold is None:
        print("mAP_list: ", mAP_list)
        outstr += 'mAP_list: ' + str(mAP_list) + '\n'

    outstr += '\n'

    ## edge-list
    if use_edgelist:
        mAP_edg = np.mean(APs_edg)
        mAP_list_edg = np.mean(APs_edg, axis=0)
        if iou_threshold is None:
            iou_str = '@[0.5:0.95]'
        else:
            iou_str = '@[' + str(iou_threshold) + ']'

        print(epochs, "epochs model, ", "iou_threshold: ", iou_str, ' #original_mask + edge-list')
        print("mAP_edg: ", mAP_edg)
        outstr += '## ' + dataset_type + ' datas\n'
        outstr += epochs + ' epochs model,' + ' iou_threshold: ' + iou_str + ' #original_mask + edge-list' + '\n'
        outstr += 'mAP_edg: ' + str(mAP_edg) + '\n'

        if iou_threshold is None:
            print("mAP_list_edg: ", mAP_list_edg)
            outstr += 'mAP_list_edg: ' + str(mAP_list_edg) + '\n'

        outstr += '\n'

    # write validation result to txt
    write_path = os.path.join(eval_result_save_dir, 'eval_result.txt')
    fp = open(write_path, 'a')
    fp.write(outstr)
    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_basedir', '-db', type=str, default='../data', help="set the data base dir")
    parser.add_argument('--dataset', '-ds', type=str, choices=['val', 'test'],
                        default='val', help="choose a dataset")
    parser.add_argument('--epochs', '-ep', type=str, default='0100', help="the epochs of trained model")
    parser.add_argument('--use_edgelist', '-el', type=int, choices=[0, 1],
                        default=1, help="use edgelist or not")
    args = parser.parse_args()

    run_params = {
        "data_base_dir": args.data_basedir,
        "dataset_type": args.dataset,
        "epochs": args.epochs,
        "use_edgelist": args.use_edgelist == 1,
    }

    segment_evaluate(**run_params)
