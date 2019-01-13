import tensorflow as tf

############################################
#           dataset params
############################################
tf.app.flags.DEFINE_integer(
    'nSketchClasses', 47,
    'The number of sketch classes.')

tf.app.flags.DEFINE_integer(
    'nTrainImgs', 5617,
    'The number of sketch classes.')

tf.app.flags.DEFINE_integer(
    'nValImgs', 535,
    'The number of sketch classes.')

tf.app.flags.DEFINE_integer(
    'nTestImgs', 1113,
    'The number of sketch classes.')

############################################
#      common dir / filepath / params
############################################
tf.app.flags.DEFINE_float(
    'mean', (104.00698793, 116.66876762, 122.67891434), 'The mean array')

tf.app.flags.DEFINE_string(
    'outputs_base_dir', 'outputs',
    'The base folder of outputs')

tf.app.flags.DEFINE_string(
    'snapshot_folder_name', 'snapshot',
    'The folder of trained model')

tf.app.flags.DEFINE_string(
    'data_base_dir',
    '../data',
    'The folder of train_images')

############################################
#   sketch segmentation training parameter
############################################

tf.app.flags.DEFINE_string(
    'resnet_pretrained_model_path',
    'resnet_pretrained_model/ResNet101_init.tfmodel',
    'The pre_trained resnet model path for training')

tf.app.flags.DEFINE_string(
    'log_folder_name', 'log',
    'The log dir during training')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.0001, '0.0001 for first train.')

tf.app.flags.DEFINE_float(
    'learning_rate_end', 0.00001, '0.00001 for first train.')

tf.app.flags.DEFINE_integer(
    'max_iteration', 100000, 'The max_iteration of training.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The optimizer used to train. Choose from [sgd, mom, adam]')

tf.app.flags.DEFINE_string(
    'upsample_mode', 'deconv',
    'The upsample mode of resizing to image_size. Choose from [bilinear, deconv]')

tf.app.flags.DEFINE_boolean(
    'data_aug', False, 'Use data augmentation during training.')

tf.app.flags.DEFINE_boolean(
    'image_down_scaling', False, 'Down Scaling input image when training.')

tf.app.flags.DEFINE_boolean(
    'ignore_class_bg', True, 'Whether to ignore background class.')

tf.app.flags.DEFINE_integer(
    'summary_write_freq', 50, 'Write summary frequence.')

tf.app.flags.DEFINE_integer(
    'save_model_freq', 20000, 'Save model frequence.')

tf.app.flags.DEFINE_integer(
    'count_left_time_freq', 100, 'Count left time frequence.')


FLAGS = tf.app.flags.FLAGS
