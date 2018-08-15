import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from layers import encode, transform, decode, ndf, lReLU


def generator(name, in_data, training=True):
    '''生成器，通过name来获得具体的生成器'''
    # 1050 gtx 2G 约100ms一幅(256x256x3)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        features = encode(in_data, training)
        transformed_features = transform(features, training)
        genImg = decode(transformed_features, training)
        return genImg


probe = None


def discriminator(name, in_data, training=True):
    '''通过name获得具体的辨别器'''
    # 1050 gtx 2G 约68ms一幅(256x256x3)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        l1 = tf.layers.conv2d(
            in_data, ndf, 3, 2, 'same', activation=lReLU)  # 128x128x64
        l2 = tf.layers.conv2d(
            l1, ndf, 3, 2, 'same', activation=lReLU)  # 64x64x64
        l1x1 = tf.layers.conv2d(
            l2, 32, 1, 1, activation=lReLU, use_bias=False)  # 64x64x32
        bn1 = tf.layers.batch_normalization(l1x1, training=training)
        l3 = tf.layers.average_pooling2d(bn1, 2, 2, 'same')  # 32x32x32

        l4 = tf.layers.conv2d(
            l3, ndf * 2, 3, 2, 'same', activation=lReLU)  # 16x16x128
        l5 = tf.layers.conv2d(
            l4, ndf * 2, 3, 2, 'same', activation=lReLU)  # 8x8x128
        l1x1 = tf.layers.conv2d(
            l5, 32, 1, 1, activation=lReLU, use_bias=False)  # 8x8x32
        bn2 = tf.layers.batch_normalization(l1x1, training=training)
        l6 = tf.layers.average_pooling2d(bn2, 2, 2, 'same')  # 4x4x32

        l7 = tf.layers.conv2d(
            l6, ndf * 4, 3, 2, 'same', activation=lReLU, name='l7')  # 2x2x256
        l8 = tf.layers.conv2d(
            l7, ndf * 4, 3, 2, 'same', activation=lReLU,
            use_bias=False)  # 1x1x256
        bn3 = tf.layers.batch_normalization(l8, training=training)
        flat = tf.layers.flatten(bn3)
        den = tf.layers.dense(flat, 32, activation=lReLU, use_bias=False)
        bn4 = tf.layers.batch_normalization(den, training=training)
        # global probe
        # probe = bn4
        out = tf.layers.dense(bn4, 1, activation=tf.sigmoid)

        return out


class CycleGan:
    '''构建 cycle gan 网络'''

    def __init__(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.input_real_A = tf.placeholder(
            tf.float32, [None, 256, 256, 3], name='input_A')
        self.input_real_B = tf.placeholder(
            tf.float32, [None, 256, 256, 3], name='input_B')
        self.build_fake()
        self.build_disc()
        self.build_loss()
        self.build_opt()

    def build_fake(self):
        self.genB = generator('genA2B', self.input_real_A)
        self.genA = generator('genB2A', self.input_real_B)
        self.fakeB = generator('genA2B', self.genA)
        self.fakeA = generator('genB2A', self.genB)

    def build_disc(self):
        self.discRealA = discriminator('discA', self.input_real_A)
        self.discRealB = discriminator('discB', self.input_real_B)
        self.discGenA = discriminator('discA', self.genA)
        self.discGenB = discriminator('discB', self.genB)

    def build_loss(self):
        '''
        1. disc约束：对于disc，discReal尽可能接近1， discFake尽可能接近0
        2. gen约束：对于gen，discFake尽可能接近1
        3. cyclic约束：fakeA尽可能接近input_real_A，fakeB尽可能接近input_real_B
        '''

        # 辨别器A的loss
        discA_loss_1 = tf.reduce_mean(
            tf.squared_difference(self.discRealA, 1), name='discA_loss_1')
        discA_loss_2 = tf.reduce_mean(
            tf.squared_difference(self.discGenA, 0), name='discA_loss_2')
        self.discA_loss = tf.multiply((discA_loss_1 + discA_loss_2), 0.5,
                                      'discA_loss')

        # 辨别器B的loss
        discB_loss_1 = tf.reduce_mean(
            tf.squared_difference(self.discRealB, 1), name='discB_loss_1')
        discB_loss_2 = tf.reduce_mean(
            tf.squared_difference(self.discGenB, 0), name='discB_loss_2')
        self.discB_loss = tf.multiply((discB_loss_1 + discB_loss_2), 0.5,
                                      'discB_loss')

        # 生成器 loss part I
        genA2B_loss_1 = tf.reduce_mean(
            tf.squared_difference(self.discGenB, 1), name='genA2B_loss_1')
        genB2A_loss_1 = tf.reduce_mean(
            tf.squared_difference(self.discGenA, 1), name='genB2A_loss_1')

        # 循环约束，生成器 loss part II
        cyc_A_loss = tf.reduce_mean(
            tf.abs(self.input_real_A - self.fakeA), name='cycA_loss')
        cyc_B_loss = tf.reduce_mean(
            tf.abs(self.input_real_B - self.fakeB), name='cycB_loss')
        cyc_loss = tf.add(cyc_A_loss, cyc_B_loss, name='cyclic_loss')

        # 生成器的loss
        self.genA2B_loss = tf.add(
            genA2B_loss_1, 10.0 * cyc_loss, name='genA2B_loss')
        self.genB2A_loss = tf.add(
            genB2A_loss_1, 10.0 * cyc_loss, name='ganB2A_loss')

    def build_opt(self):
        '''优化器'''

        all_vs = tf.trainable_variables()
        genA2B_vars = [v for v in all_vs if v.name.startswith('genA2B')]
        genB2A_vars = [v for v in all_vs if v.name.startswith('genB2A')]
        discA_vars = [v for v in all_vs if v.name.startswith('discA')]
        discB_vars = [v for v in all_vs if v.name.startswith('discB')]

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.opt_discA = tf.train.AdamOptimizer(
                name='opt_discA', learning_rate=self.lr).minimize(
                    self.discA_loss, var_list=discA_vars)
            self.opt_discB = tf.train.AdamOptimizer(
                name='opt_discB', learning_rate=self.lr).minimize(
                    self.discB_loss, var_list=discB_vars)
            self.opt_genA2B = tf.train.AdamOptimizer(
                name='opt_genA2B', learning_rate=self.lr).minimize(
                    self.genA2B_loss, var_list=genA2B_vars)
            self.opt_genB2A = tf.train.AdamOptimizer(
                name='opt_genB2A', learning_rate=self.lr).minimize(
                    self.genB2A_loss, var_list=genB2A_vars)

    def train_on_batch(self, sess, feed_dict):
        _, d_A_loss = sess.run(
            [self.opt_discA, self.discA_loss], feed_dict=feed_dict)
        _, d_B_loss = sess.run(
            [self.opt_discB, self.discB_loss], feed_dict=feed_dict)
        _, g_A2B_loss = sess.run(
            [self.opt_genA2B, self.genA2B_loss], feed_dict=feed_dict)
        _, g_B2A_loss = sess.run(
            [self.opt_genB2A, self.genB2A_loss], feed_dict=feed_dict)
        return g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss

    def train(self, data_A_paths, data_B_paths, batch_size=1, epoch=100):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ii in range(epoch):
                idxA = np.random.choice(len(data_A_paths), batch_size)
                idxB = np.random.choice(len(data_B_paths), batch_size)

                data_A = np.array(
                    [plt.imread(data_A_paths[idx]) for idx in idxA])
                data_B = np.array(
                    [plt.imread(data_B_paths[idx]) for idx in idxB])

                data_A = data_A / 127.5 - 1
                data_B = data_B / 127.5 - 1

                feed_dict = {
                    self.input_real_A: data_A,
                    self.input_real_B: data_B,
                    self.lr: 1e-4
                }
                # 以上耗时约 224ms

                losses = self.train_on_batch(sess, feed_dict)
                print('gA2Bl: {}, gB2Al: {}, dAl: {}, dBl: {}'.format(
                    *np.round(losses, 3)))
