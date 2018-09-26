import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import os


def sample_z(m, n):
    return np.random.uniform(-1, 1, size=[m, n])


def generator(z, isTrain, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        dense1 = tf.layers.dense(z, 128, activation=tf.nn.relu, kernel_initializer=w_init)
        o = tf.layers.dense(dense1, 784, activation=tf.nn.tanh, kernel_initializer=w_init)
        return o


def discriminator(x, isTrain, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        dense1 = tf.layers.dense(x, 128, activation=tf.nn.relu, kernel_initializer=w_init)
        dense2 = tf.layers.dense(dense1, 1, kernel_initializer=w_init)
        o = tf.nn.sigmoid(dense2)
        return o, dense2


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    return fig


z = tf.placeholder(tf.float32, shape=[None, 100], name='z')
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
isTrain = tf.placeholder(dtype=tf.bool)
G_sample = generator(z, isTrain)

D_real, D_logit_real = discriminator(x, isTrain)
D_fake, D_logit_fake = discriminator(G_sample, isTrain, reuse=True)
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))  # TODO

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]
D_solver = tf.train.AdamOptimizer(0.001).minimize(D_loss, var_list=D_vars)
G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=G_vars)

batch_size = 128
z_dim = 100
mnist = input_data.read_data_sets('mnist/', one_hot=True)
i = 0

if not os.path.exists('output/GAN/'):
    os.makedirs('output/GAN/')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(100000):
        x_batch, _ = mnist.train.next_batch(batch_size)

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={x: x_batch, z: sample_z(batch_size, z_dim), isTrain: True}
        )
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={z: sample_z(batch_size, z_dim), isTrain: True}
        )

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})
            fig = plot(samples)
            plt.savefig('output/GAN/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            i += 1

            print('[{:4}] D_loss: {:.4} G_loss: {:.4}'.format(
                str(i).zfill(3), D_loss_curr, G_loss_curr))
