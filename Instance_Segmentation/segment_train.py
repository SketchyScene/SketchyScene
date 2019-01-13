import os
import sys
import tensorflow as tf
import argparse
from SketchDataset import SketchDataset

sys.path.append('libs')
from config import Config
import model as modellib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))


class SketchTrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sketchyscene"

    # Batch size is (GPU_COUNT * IMAGES_PER_GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

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

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500
    TOTAL_EPOCH = 100

    # Optimizer
    # choose from ['sgd', 'adam']
    TRAIN_OPTIMIZER = 'sgd'

    LEARNING_RATE = 0.0001

    # When training, only use the pixels with value 1 in target_mask to contribute to the loss.
    IGNORE_BG = True


def instance_segment_train(**kwargs):
    data_base_dir = kwargs['data_base_dir']
    init_with = kwargs['init_with']

    outputs_base_dir = 'outputs'
    pretrained_model_base_dir = 'pretrained_model'

    save_model_dir = os.path.join(outputs_base_dir, 'snapshot')
    log_dir = os.path.join(outputs_base_dir, 'log')
    coco_model_path = os.path.join(pretrained_model_base_dir, 'mask_rcnn_coco.h5')
    imagenet_model_path = os.path.join(pretrained_model_base_dir, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config = SketchTrainConfig()
    config.display()

    # Training dataset
    dataset_train = SketchDataset(data_base_dir)
    dataset_train.load_sketches("train")
    dataset_train.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=save_model_dir, log_dir=log_dir)

    if init_with == "imagenet":
        print("Loading weights from ", imagenet_model_path)
        model.load_weights(imagenet_model_path, by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        print("Loading weights from ", coco_model_path)
        model.load_weights(coco_model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        last_model_path = model.find_last()[1]
        print("Loading weights from ", last_model_path)
        model.load_weights(last_model_path, by_name=True)
    else:
        print("Training from fresh start.")

    # Fine tune all layers
    model.train(dataset_train,
                learning_rate=config.LEARNING_RATE,
                epochs=config.TOTAL_EPOCH,
                layers="all")

    # Save final weights
    save_model_path = os.path.join(save_model_dir, "mask_rcnn_" + config.NAME + "_" + str(config.TOTAL_EPOCH) + ".h5")
    model.keras_model.save_weights(save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_basedir', '-db', type=str, default='../data', help="set the data base dir")
    parser.add_argument('--init_model', '-init', type=str, choices=['imagenet', 'coco', 'last', 'none'],
                        default='none', help="choose a initial pre-trained model")
    args = parser.parse_args()

    run_params = {
        "data_base_dir": args.data_basedir,
        "init_with": args.init_model
    }

    instance_segment_train(**run_params)