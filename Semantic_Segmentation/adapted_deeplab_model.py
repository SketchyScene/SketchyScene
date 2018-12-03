# This code is modified from
# https://github.com/tensorflow/models/blob/master/research/resnet/resnet_model.py
# and
# https://github.com/chenxi116/TF-deeplab/blob/master/deeplab_model.py

from tensorflow.python.training import moving_averages
import numpy as np
import tensorflow as tf

import sys

sys.path.append('libs')
import tensorflow_util as tfutils


def myConvertFun(tensor_):
    tensor_selected = tensor_[:, :, :, 1:]
    tensor_selected = np.squeeze(tensor_selected)
    return tensor_selected


class DeepLab(object):
    """DeepLab model."""

    def __init__(self, batch_size=1,
                 num_classes=47,
                 lrn_rate=0.0001,
                 lr_decay_step=70000,
                 lrn_rate_end=0.00001,
                 lrn_rate_decay_rate=0.7,
                 num_residual_units=[3, 4, 23, 3],
                 weight_decay_rate=0.0001,
                 relu_leakiness=0.0,
                 bn=False,
                 filters=[64, 256, 512, 1024, 2048],
                 optimizer='adam',  # 'sgd' or 'mom' or 'adam'
                 images=tf.placeholder(tf.float32),
                 labels=tf.placeholder(tf.int32),
                 upsample_mode='deconv',  # 'bilinear' or 'deconv'
                 data_aug=False,
                 data_aug_scale_low=0.6,
                 data_aug_scale_up=1.1,
                 image_down_scaling=False,
                 ignore_class_bg=True,
                 mode='test'):
        """DeepLab constructor.

    Args:
      : Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, image_size, image_size]
    """
        self.images = images
        self.labels = labels
        self.H = tf.shape(self.images)[1]
        self.W = tf.shape(self.images)[2]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lrn_rate = lrn_rate
        self.lr_decay_step = lr_decay_step
        self.lrn_rate_end = lrn_rate_end
        self.lrn_rate_decay_rate = lrn_rate_decay_rate
        self.num_residual_units = num_residual_units
        self.weight_decay_rate = weight_decay_rate
        self.relu_leakiness = relu_leakiness
        self.bn = bn
        self.filters = filters
        self.optimizer = optimizer
        self.upsample_mode = upsample_mode
        self.data_aug = data_aug
        self.data_aug_scale_low = data_aug_scale_low
        self.data_aug_scale_up = data_aug_scale_up
        self.image_down_scaling = image_down_scaling
        self.ignore_class_bg = ignore_class_bg
        self.mode = mode
        self._extra_train_ops = []

        with tf.variable_scope("ResNet"):
            self.build_graph()

    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('group_1'):
            x = self.images

            ## down_scaling image: scale 0.5
            if self.image_down_scaling:
                ori_H = tf.cast(self.H, tf.float32)
                ori_W = tf.cast(self.W, tf.float32)
                scaled_H = tf.cast(tf.multiply(ori_H, 0.5), tf.int32)
                scaled_W = tf.cast(tf.multiply(ori_W, 0.5), tf.int32)

                x = tf.image.resize_bilinear(x, [scaled_H, scaled_W])

            ## data aug: scale input from 0.5-1.5
            if self.data_aug:
                data_aug_scale = tf.random_uniform([], self.data_aug_scale_low, self.data_aug_scale_up)

                ori_H = tf.cast(self.H, tf.float32)
                ori_W = tf.cast(self.W, tf.float32)
                scaled_H = tf.cast(tf.multiply(ori_H, data_aug_scale), tf.int32)
                scaled_W = tf.cast(tf.multiply(ori_W, data_aug_scale), tf.int32)
                self.H = scaled_H
                self.W = scaled_W

                x = tf.image.resize_bilinear(x, [scaled_H, scaled_W])

            x = self._conv('conv1', x, 7, 3, 64, self._stride_arr(2))
            x = self._batch_norm('bn_conv1', x)
            x = self._relu(x, self.relu_leakiness)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

        res_func = self._bottleneck_residual
        filters = self.filters

        with tf.variable_scope('group_2_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(1))
        for i in range(1, self.num_residual_units[0]):
            with tf.variable_scope('group_2_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1))

        with tf.variable_scope('group_3_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(2))
        for i in range(1, self.num_residual_units[1]):
            with tf.variable_scope('group_3_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1))

        with tf.variable_scope('group_4_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(1), 2)
        for i in range(1, self.num_residual_units[2]):
            with tf.variable_scope('group_4_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), 2)

        with tf.variable_scope('group_5_0'):
            x = res_func(x, filters[3], filters[4], self._stride_arr(1), 4)
        for i in range(1, self.num_residual_units[3]):
            with tf.variable_scope('group_5_%d' % i):
                if i == self.num_residual_units[3] - 1:
                    x = res_func(x, filters[4], filters[4], self._stride_arr(1), 4, True)
                else:
                    x = res_func(x, filters[4], filters[4], self._stride_arr(1), 4)

        with tf.variable_scope('group_last'):
            x = self._relu(x, self.relu_leakiness)
            self.res5c = x

        with tf.variable_scope('fc_final_sketch46'):
            x0 = self._conv('conv0', x, 3, filters[4], self.num_classes, self._stride_arr(1), 6, True)
            x1 = self._conv('conv1', x, 3, filters[4], self.num_classes, self._stride_arr(1), 12, True)
            x2 = self._conv('conv2', x, 3, filters[4], self.num_classes, self._stride_arr(1), 18, True)
            x3 = self._conv('conv3', x, 3, filters[4], self.num_classes, self._stride_arr(1), 24, True)
            x = tf.add(x0, x1)
            x = tf.add(x, x2)
            x = tf.add(x, x3)
            self.logits = x  # shape = [1, H/8, W/8, nClasses]

            if self.upsample_mode == 'bilinear':
                logits_up = tf.image.resize_bilinear(self.logits, [self.H, self.W])
                self.logits_up = logits_up  # shape = [1, H, W, nClasses]

            elif self.upsample_mode == 'deconv':
                W_up = tfutils.weight_variable([16, 16, self.num_classes, self.num_classes], name="W_up")
                b_up = tfutils.bias_variable([self.num_classes], name="b_up")
                up_stride = 16 if self.image_down_scaling else 8
                logits_up \
                    = tfutils.conv2d_transpose_strided(self.logits, W_up, b_up,
                                                       output_shape=[1, self.H, self.W, self.num_classes],
                                                       stride=up_stride)
                self.logits_up = logits_up  # shape = [1, H, W, nClasses]
            else:
                raise NameError("Unknown upsample mode: %s!" % self.upsample_mode)

            # logits_up_selected = tf.py_func(myConvertFun, [self.logits_up], [tf.float32])
            # self.logits_up = tf.convert_to_tensor(logits_up_selected, name='logits_up_selected')
            # logits_flat = tf.reshape(self.logits_up, [-1, self.num_classes - 1])

            logits_flat = tf.reshape(self.logits_up, [-1, self.num_classes])
            pred = tf.nn.softmax(logits_flat)
            self.pred = tf.reshape(pred, tf.shape(self.logits_up))  # shape = [1, H, W, nClasses]

            pred_label = tf.argmax(self.pred, 3)  # shape = [1, H, W]
            pred_label = tf.expand_dims(pred_label, axis=3)
            self.pred_label = pred_label  # shape = [1, H, W, 1], contains [0, nClasses)

    def _build_train_op(self):
        """Build training specific ops for the graph."""

        logits_flatten = tf.reshape(self.logits_up, [-1, self.num_classes])
        pred_flatten = tf.reshape(self.pred, [-1, self.num_classes])

        if self.data_aug:
            label_ex = tf.expand_dims(self.labels, axis=3)  # shape = [1, H, W, 1]
            label_scaled = tf.image.resize_nearest_neighbor(label_ex, [self.H, self.W])  # shape = [1, H, W, 1]
            labels_gt = tf.squeeze(label_scaled, axis=3)  # shape = [1, H, W]
        else:
            labels_gt = self.labels

        if self.ignore_class_bg:
            # ignore background labels: 255
            gt_labels_flatten = tf.reshape(labels_gt, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(gt_labels_flatten, self.num_classes - 1)), 1)
            remain_logits = tf.gather(logits_flatten, indices)
            remain_pred = tf.gather(pred_flatten, indices)
            remain_labels = tf.gather(gt_labels_flatten, indices)
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=remain_logits, labels=remain_labels)
        else:
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_up, labels=labels_gt)

        self.cls_loss = tf.reduce_mean(xent, name='xent')  # xent.shape=[nIgnoredBgPixels]
        self.cost = self.cls_loss + self._decay()
        tf.summary.scalar('cost', self.cost)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.lrn_rate,
                                                       self.global_step,
                                                       self.lr_decay_step,
                                                       end_learning_rate=self.lrn_rate_end,
                                                       power=0.9)
        tf.summary.scalar('learning rate', self.learning_rate)

        tvars = tf.trainable_variables()

        if self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise NameError("Unknown optimizer type %s!" % self.optimizer)

        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {}
        for var in tvars:
            if var.op.name.find(r'fc_final_sketch46') > 0 and var.op.name.find(r'biases') > 0:
                var_lr_mult[var] = 20.
            elif var.op.name.find(r'fc_final_sketch46') > 0:
                var_lr_mult[var] = 10.
            else:
                var_lr_mult[var] = 1.
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                          for g, v in grads_and_vars]

        ## summary grads
        # for grad, grad_var in grads_and_vars:
        #     if grad is not None:
        #         tf.summary.histogram(grad_var.op.name + "/gradient", grad)

        apply_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_step = tf.group(*train_ops)

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            factor = tf.get_variable(
                'factor', 1, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            if self.bn:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                # inv_factor = tf.reciprocal(factor)
                inv_factor = tf.div(1., factor)
                mean = tf.multiply(inv_factor, mean)
                variance = tf.multiply(inv_factor, variance)

                # tf.summary.histogram(mean.op.name, mean)
                # tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _bottleneck_residual(self, x, in_filter, out_filter, stride, atrous=1, keep_feat=False):
        """Bottleneck residual unit with 3 sub layers."""

        orig_x = x

        with tf.variable_scope('block_1'):
            x = self._conv('conv', x, 1, in_filter, out_filter / 4, stride, atrous)
            x = self._batch_norm('bn', x)
            x = self._relu(x, self.relu_leakiness)

        with tf.variable_scope('block_2'):
            x = self._conv('conv', x, 3, out_filter / 4, out_filter / 4, self._stride_arr(1), atrous)
            x = self._batch_norm('bn', x)
            x = self._relu(x, self.relu_leakiness)

            if keep_feat:
                self.feat_visual = x

        with tf.variable_scope('block_3'):
            x = self._conv('conv', x, 1, out_filter / 4, out_filter, self._stride_arr(1), atrous)
            x = self._batch_norm('bn', x)

        with tf.variable_scope('block_add'):
            if in_filter != out_filter:
                orig_x = self._conv('conv', orig_x, 1, in_filter, out_filter, stride, atrous)
                orig_x = self._batch_norm('bn', orig_x)
            x += orig_x
            x = self._relu(x, self.relu_leakiness)

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, atrous=1, bias=False):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            w = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            if atrous == 1:
                conv = tf.nn.conv2d(x, w, strides, padding='SAME')
            else:
                assert (strides == self._stride_arr(1))
                conv = tf.nn.atrous_conv2d(x, w, rate=atrous, padding='SAME')
            if bias:
                b = tf.get_variable('biases', [out_filters], initializer=tf.constant_initializer())
                return conv + b
            else:
                return conv

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.batch_size, -1])
        w = tf.get_variable(
            'DW', [self.filters[-1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _fully_convolutional(self, x, out_dim):
        """FullyConvolutional layer for final output."""
        w = tf.get_variable(
            'DW', [1, 1, self.filters[-1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.conv2d(x, w, self._stride_arr(1), padding='SAME') + b

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.expand_dims(tf.expand_dims(tf.reduce_mean(x, [1, 2]), 0), 0)
