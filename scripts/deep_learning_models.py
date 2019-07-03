#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.neural_network import BernoulliRBM
from tensorflow_probability import distributions
from tqdm import tqdm


class RBM:

    def __init__(self, images, n_components, learning_rate, batch_size, n_iter, random_state):
        """
        :param images: input data for the RBM neural network
        :param n_components: number of hidden units for the RBM neural network
        :param learning_rate: learning rate for the RBM neural network
        :param batch_size: batch size for the RBM neural network
        :param n_iter: number of iterations/epochs for the RBM neural network
        :param random_state: random state for the RBM neural network
        """
        self.images = images
        self.batch_size = batch_size
        self.epochs = n_iter
        self.x = 0
        self.rbm = BernoulliRBM(
            n_components=n_components,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_iter=self.epochs,
            random_state=random_state,
            verbose=1)

    def fit(self):
        """
        :return: void
        """
        self.x, _ = self.images.train.next_batch(self.batch_size)
        self.rbm.fit(self.x)

    def gibbs_sampling(self, k):
        """
        :param k: number of steps of Gibbs sampling
        :return: void
        """
        for i in range(k):
            gibbs_x = self.rbm.gibbs(self.x)
            self.x = np.zeros_like(self.x)
            self.x[gibbs_x] = 1

    def generate_images(self, num_hidden_nodes):
        """
        :param num_hidden_nodes: number of hidden nodes/units
        :return: void
        """
        plt.figure(figsize=(6, 6))
        for i, comp in enumerate(self.x):
            plt.subplot(10, 10, i + 1)
            plt.imshow(comp.reshape((28, 28)), cmap="gray", interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle("RBM reconstructed image with " + str(num_hidden_nodes) + " hidden nodes", fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig("RBM reconstructed image with " + str(num_hidden_nodes) + " hidden nodes.png")


class VAE:

    def __init__(self, images, code_units):
        """
        :param images: input data for the Variational Auto Encoder
        :param code_units: number of code units for the VAE
        """
        tf.reset_default_graph()
        self.images = images
        self.data = tf.placeholder(tf.float32, [None, 28, 28])
        self.code_units = code_units
        posterior = self.encode()
        self.sample_r = posterior.sample()
        prior = self.make_prior()

        likelihood = self.decode(
            self.sample_r, [28, 28]).log_prob(self.data)

        divergence = distributions.kl_divergence(posterior, prior)
        self.evidence = tf.reduce_mean(likelihood - divergence)
        self.optimize = tf.train.AdamOptimizer(0.001).minimize(-self.evidence)

        self.samples = self.decode(
            prior.sample(10), [28, 28]).mean()

    def encode(self):
        """
        :return: void
        """
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            mean = tf.layers.dense(
                tf.layers.dense(
                    tf.layers.dense(
                        tf.layers.flatten(self.data), 784, tf.nn.relu
                    ), 256, tf.nn.relu
                ), self.code_units
            )

            variance = tf.layers.dense(
                tf.layers.dense(
                    tf.layers.dense(
                        tf.layers.flatten(self.data), 784, tf.nn.relu
                    ), 256, tf.nn.relu
                ), self.code_units, tf.nn.softplus
            )

            return distributions.MultivariateNormalDiag(mean, variance)

    def make_prior(self):
        """
        :return: void
        """
        mean = tf.zeros(self.code_units)
        variance = tf.ones(self.code_units)
        return distributions.MultivariateNormalDiag(mean, variance)

    @staticmethod
    def decode(code, data_shape):
        """
        :param code: number of code units
        :param data_shape: dimensionality of the input data
        :return: void
        """
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            logit = tf.reshape(
                tf.layers.dense(
                    tf.layers.dense(
                        tf.layers.dense(code, 256, tf.nn.relu), 784, tf.nn.relu
                    ), np.prod(data_shape)
                ), [-1] + data_shape
            )

            return distributions.Independent(distributions.Bernoulli(logit), 2)

    def generate_images(self, epochs):
        """
        :param epochs: number of epochs
        :return: void
        """
        with tf.train.MonitoredSession() as sess:
            for epoch in range(epochs):
                plt.clf()
                feed = {self.data:
                            self.images.test.images.reshape([-1, 28, 28])}

                test_evidence, test_codes, test_samples = sess.run(
                    [self.evidence, self.sample_r, self.samples], feed)

                print('\nEpoch ' + str(epoch + 1) + ' for ' + \
                      str(self.code_units) + ' code units')
                print("Evidence lower bound: ", test_evidence)

                for _ in range(self.images.train.num_examples // 100):
                    feed = {self.data:
                        self.images.train.next_batch(100)[0].reshape(
                            [-1, 28, 28])}
                    sess.run(self.optimize, feed)

                fig, axes = plt.subplots(
                    figsize=(20, 4),
                    nrows=1,
                    ncols=10,
                    sharey=True,
                    sharex=True
                )

                for ax, img in zip(axes.flatten(), test_samples[0:]):
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.imshow(img.reshape((28, 28)), cmap="gray")
                fig.tight_layout(pad=0.1)
                fig.suptitle('Epoch ' + str(epoch + 1), fontsize=16)
                plt.savefig('VAE ' + str(self.code_units) + ' code units - ' \
                            + str(epoch + 1) + ' Epochs.png')
                # plt.show()


class VAEConvolutionNeuralNet:

    def __init__(self, images, input_shape, output_shape):
        """
        :param images: input data for the Convolutional VAE
        :param input_shape: dimensionality of the input data
        :param output_shape: dimensionality of the output data
        """
        self.images = images
        inputmatx, inputmaty = input_shape
        outputmatx, outputmaty = output_shape
        self.inputTensor = tf.placeholder(
            tf.float32,
            [None, inputmatx, inputmaty, 1]
        )
        self.outputTensor = tf.placeholder(
            tf.float32,
            [None, outputmatx, outputmaty, 1]
        )
        self.encoder_out = 0
        self.decoder_out = 0
        self.logits = 0
        self.cost = 0
        self.optimizer = 0

    def encode(self):
        """
        :return: void
        """
        self.encoder_out = tf.layers.conv2d(
            self.inputTensor,
            64,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.encoder_out = tf.layers.max_pooling2d(
            self.encoder_out,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )

        self.encoder_out = tf.layers.conv2d(
            self.encoder_out,
            32,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.encoder_out = tf.layers.max_pooling2d(
            self.encoder_out,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )

        self.encoder_out = tf.layers.conv2d(
            self.encoder_out,
            16,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.encoder_out = tf.layers.max_pooling2d(
            self.encoder_out,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )

    def decode(self):
        """
        :return: void
        """
        self.decoder_out = tf.image.resize_nearest_neighbor(
            self.encoder_out, (7, 7)
        )

        self.decoder_out = tf.layers.conv2d(
            self.decoder_out,
            16,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.decoder_out = tf.image.resize_nearest_neighbor(
            self.decoder_out, (14, 14)
        )

        self.decoder_out = tf.layers.conv2d(
            self.decoder_out,
            32,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.decoder_out = tf.image.resize_nearest_neighbor(
            self.decoder_out, (28, 28)
        )

        self.decoder_out = tf.layers.conv2d(
            self.decoder_out,
            64,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )

        self.logits = tf.layers.conv2d(
            self.decoder_out,
            1,
            kernel_size=(3, 3),
            padding='same',
            activation=None
        )

        self.decoder_out = tf.nn.sigmoid(self.logits)

    def compile_(self):
        """
        :return: void
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=self.outputTensor
        )
        self.cost = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def train(self, epochs, batch_size):
        """
        :param epochs: number of epochs
        :param batch_size: batch size for training
        :return: void
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        np.random.shuffle(self.images.test.images)
        test_images = self.images.test.images[:10]

        for epoch in range(epochs):
            print("\nEpoch " + str(epoch))
            for _ in tqdm(range(self.images.train.num_examples // batch_size),
                          total=self.images.train.num_examples // batch_size):
                batch = self.images.train.next_batch(batch_size)
                images = batch[0].reshape((-1, 28, 28, 1))
                batch_cost, _ = sess.run(
                    [self.cost, self.optimizer],
                    feed_dict={self.inputTensor: images, self.outputTensor: images}
                )

            print("Training loss after Epoch " + str(epoch + 1) + " = " + str(round(batch_cost, 3)))
            modeled_images = sess.run(
                self.decoder_out,
                feed_dict={
                    self.inputTensor: test_images.reshape((10, 28, 28, 1))
                }
            )
            self.draw_image(test_images, modeled_images, epoch)

    @staticmethod
    def draw_image(original_images, modeled_images, epoch):
        """
        :param original_images: original input image
        :param modeled_images: modeled output image of Convolutional VAE
        :param epoch: epoch number
        :return: void
        """
        plt.clf()
        f, axarr = plt.subplots(2, 10, sharex=True, sharey=True, figsize=(20, 5))
        for images, row in zip([original_images, modeled_images], axarr):
            for image, ax in zip(images, row):
                ax.imshow(image.reshape((28, 28)), cmap="gray")
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        f.tight_layout(pad=0.1)
        plt.savefig("VAE CNN Reconstruction after " + str(epoch + 1) + " epochs.png")
        # plt.show()
