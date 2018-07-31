import tensorflow as tf
import numpy as np

ngf = 32 # numbers of 1st layer gen filters
ndf = 64 # numbers of 1st layer disc filters
batch_size = 1
pool_size = 1
img_width = 256
img_height = 256
img_depth = 3

def encode (name, in_data, is_training):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    l1 = tf.contrib.layers.conv2d(in_data, ngf, 3, 2) # 128x128x32
    l2 = tf.contrib.layers.conv2d(l1, ngf*2, 3, 2) # 64x64x64
    l3 = tf.contrib.layers.conv2d(l2, ngf*4, 3, 2) # 32x32x128
    out = tf.contrib.layers.batch_norm(l3, is_training=is_training)
    return out

