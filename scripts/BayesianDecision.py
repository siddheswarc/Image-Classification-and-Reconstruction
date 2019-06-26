#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from confusion_matrix import ConfusionMatrix

confusion_matrix = ConfusionMatrix()


class DiscriminantAnalysis:
    NUMBER_OF_OUTPUT_CLASSES = 10
    NOISE = 0.1  # Noise to avoid singular matrix error

    def __init__(self):
        self.mean_digits = {}
        self.std_digits = {}
        self.w = {}
        self.n = {}
        self.b = {}
        self.yhat = []

    def find_mean_digits(self, data, dimensions):
        """
        :param data: dictionary containing images in their respective class keys
        :param dimensions: dimension of the images
        :return: mean of the image classes
        """

        for i in data:
            mean = np.zeros(dimensions)
            for j in data[i]:
                mean += j
            mean /= len(data[i])
            self.mean_digits[i] = mean

        return self.mean_digits

    def find_standard_deviation_digits(self, data, mean, dimensions):
        """
        :param data: dictionary containing images in their respective class keys
        :param mean: mean of the image classes
        :param dimensions: dimension of the images
        :return: standard deviation of the image classes
        """

        for i in data:
            std = np.zeros(dimensions)
            for j in data[i]:
                std += (j - mean[i]) ** 2
            std /= len(data[i])
            self.std_digits[i] = std

        return self.std_digits

    @staticmethod
    def save_digit_image(mean, std, digit):
        plt.clf()
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(mean, cmap="gray")
        axarr[0].set_title('Mean Digit')
        axarr[1].imshow(std, cmap="gray")
        axarr[1].set_title('Standard Deviation Digit')
        f.suptitle('Digit ' + str(digit), fontsize=16)
        plt.savefig(r'../output_images/Digit ' + str(digit) + '.jpg')

    @staticmethod
    def find_confusion_matrix(y, yhat):
        """
        :param y: actual label
        :param yhat: predicted label
        :return: confusion matrix
        """
        return confusion_matrix.calculate(y, yhat)

    def calculate_w(self, std):
        """
        :param std: output of find_std_digits()
        :return: parameter 'w' for Quadratic Discriminant Analysis
        """

        for i in range(self.NUMBER_OF_OUTPUT_CLASSES):
            std[i] = std[i].flatten()
            tmp = np.zeros((len(std[i]), len(std[i])))

            for j in range(len(std[i])):
                tmp[j][j] = (std[i][j] ** 2) + self.NOISE  # Adding noise to avoid Singular Matrix error
            self.w[i] = -0.5 * np.linalg.inv(tmp)

        return self.w

    def calculate_n(self, w, mean):
        """
        :param w: parameter 'w' for Quadratic Discriminant Analysis
        :param mean: output of find_mean_digits()
        :return: parameter 'n' for Quadratic Discriminant Analysis
        """

        for i in range(self.NUMBER_OF_OUTPUT_CLASSES):
            mean[i] = mean[i].flatten()
            self.n[i] = np.matmul(-2 * w[i], mean[i])

        return self.n

    def calculate_b(self, n, mean):
        """
        :param n: parameter 'n' for Quadratic Discriminant Analysis
        :param mean: output of find_mean_digits()
        :return: parameter 'b' for Quadratic Discriminant Analysis
        """

        for i in range(self.NUMBER_OF_OUTPUT_CLASSES):
            self.b[i] = -0.5 * (np.matmul(n[i], np.transpose(mean[i])))

        return self.b

    @staticmethod
    def find_accuracy(actual, predicted):
        ctr = 0
        for i in range(len(actual)):
            if predicted[i] == actual[i]:
                ctr += 1

        return ctr / len(actual)

    def predict(self, dataset, x_test, y_test):
        """
        :param x_test: testing feature-vector
        :param y_test: testing label
        :return: predicted label, number of correct and incorrect predictions, accuracy and error rate
        """

        print('\nTesting ' + dataset + ' using Discriminant Analysis')
        self.yhat = []
        for x in tqdm(x_test, total=len(x_test)):
            g = []
            x = x.flatten()

            for i in range(len(self.w)):
                g.append(np.matmul(x, np.matmul(self.w[i], x)) + np.matmul(np.transpose(self.n[i]), x) + self.b[i])

            g = np.asarray(g)
            self.yhat.append(np.argmax(g))

        accuracy = self.find_accuracy(y_test, self.yhat)
        confusion = self.find_confusion_matrix(y_test, self.yhat)

        return accuracy, confusion

    def fit(self, x_train, y_train):
        """
        :param x_train: training feature-vector
        :param y_train: training label
        :return: void
        """

        DIMENSIONS = x_train.shape[1:]

        data = {}
        for i in range(self.NUMBER_OF_OUTPUT_CLASSES):
            data[i] = []

        # Create dict
        for i in range(y_train.shape[0]):
            data[y_train[i]].append(x_train[i, :, :])

        self.find_standard_deviation_digits(
            data, self.find_mean_digits(data, DIMENSIONS), DIMENSIONS)

        # Save images
        for i in range(len(data)):
            self.save_digit_image(self.mean_digits[i], self.std_digits[i], i)

        self.calculate_b(
            self.calculate_n(
                self.calculate_w(self.std_digits), self.mean_digits), self.mean_digits)
