import tensorflow as tf
import numpy as np

from layers import encode, transform, decode, ndf, lReLU

def generator(name, in_data, training, reuse):
    '''生成器，通过name来获得具体的生成器'''
    # 1050 gtx 2G 约100ms一幅(256x256x3)
    with tf.variable_scope(name, reuse=reuse):
        features = encode(in_data, training)
        transformed_features = transform(features, training)
        genImg = decode(transformed_features, training)
        return genImg

def discriminator(name, in_data, training, reuse):
    '''通过name获得具体的辨别器'''
    # 1050 gtx 2G 约68ms一幅(256x256x3)
    with tf.variable_scope(name, reuse=reuse):
        l1 = tf.layers.conv2d(in_data, ndf, 3, 2, 'same', activation=lReLU) # 128x128x64
        l2 = tf.layers.conv2d(l1, ndf*2, 3, 2, 'same', activation=lReLU, use_bias=False) # 64x64x128
        bn1 = tf.layers.batch_normalization(l2, training=training)
        l3 = tf.layers.average_pooling2d(bn1, 2, 2, 'same') # 32x32x128
        
        l4 = tf.layers.conv2d(l3, ndf*2, 3, 2, 'same', activation=lReLU) # 16x16x128
        l5 = tf.layers.conv2d(l4, ndf*4, 3, 2, 'same', activation=lReLU, use_bias=False) # 8x8x256
        bn2 = tf.layers.batch_normalization(l5, training=training)
        l6 = tf.layers.average_pooling2d(bn2, 2, 2, 'same') # 4x4x256

        l7 = tf.layers.conv2d(l6, ndf*4, 3, 2, 'same', activation=lReLU) # 2x2x256
        l8 = tf.layers.conv2d(l7, ndf*8, 3, 2, 'same', activation=lReLU, use_bias=False) # 1x1x512
        bn3 = tf.layers.batch_normalization(l8, training=training)
        flat = tf.layers.flatten(bn3)
        out = tf.layers.dense(flat, 1, activation=tf.sigmoid)

        return out

# gened_A = generator()