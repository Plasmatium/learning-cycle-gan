import tensorflow as tf
import numpy as np

ngf = 32  # numbers of 1st layer gen filters
ndf = 64  # numbers of 1st layer disc filters
batch_size = 1
pool_size = 1
img_width = 256
img_height = 256
img_depth = 3
'''
BN层操作

URL：https://tensorflow.google.cn/api_docs/python/tf/layers/batch_normalization

Note: when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op. Also, be sure to add any batch_normalization ops before getting the update_ops collection. Otherwise, update_ops will be empty, and training/inference will not work properly. For example:

  x_norm = tf.layers.batch_normalization(x, training=training)

  # ...

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

'''


def lReLU(x, alpha=0.2):
    return tf.maximum(x * alpha, x)


def encode(in_data, training):
    '''从原图获取特征'''
    l1 = tf.layers.conv2d(
        in_data, ngf * 2, 3, 2, 'same', activation=lReLU)  # 128x128x64
    l2 = tf.layers.conv2d(
        l1, ngf * 4, 3, 2, 'same', activation=lReLU)  # 64x64x128
    l1x1 = tf.layers.conv2d(
        l2, 32, 1, 1, activation=lReLU)  # 64x64x32
    l3 = tf.layers.conv2d(
        l1x1, ngf * 8, 2, 1, 'same', activation=lReLU,
        use_bias=False)  # 64x64x256
    out = tf.layers.batch_normalization(l3, training=training)
    return out


def rn_block(in_data, num_features):
    ''' build resnet block'''
    l1 = tf.layers.conv2d(
        in_data, num_features, 3, 1, 'same', activation=lReLU)
    l2 = tf.layers.conv2d(l1, num_features, 3, 1, 'same', activation=lReLU)
    return (l2 + in_data)


def transform(in_data, training):
    '''将特征做变换'''
    l1 = rn_block(in_data, 64 * 4)  # 64x64x256
    l2 = rn_block(l1, 64 * 4)
    l3 = rn_block(l2, 64 * 4)
    bn = tf.layers.batch_normalization(l3, training=training)
    l4 = rn_block(bn, 64 * 4)
    l5 = rn_block(l4, 64 * 4)
    l6 = rn_block(l5, 64 * 4)  # 64x64x256
    out = tf.layers.batch_normalization(l6, training=training)
    return out


def decode(in_data, training):
    '''将变换后的特征复原到原图'''
    l1 = tf.layers.conv2d_transpose(
        in_data, ngf * 2, 3, 2, 'same', activation=lReLU)
    l2 = tf.layers.conv2d_transpose(l1, ngf, 3, 2, 'same', activation=lReLU)
    l3 = tf.layers.conv2d(
        l2, 3, 2, 1, 'same', activation=lReLU, use_bias=False)
    logits = tf.layers.batch_normalization(l3, training=training)
    out = tf.tanh(logits)
    return out
