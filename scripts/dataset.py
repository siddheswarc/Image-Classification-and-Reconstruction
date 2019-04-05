#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import os

import numpy as np
import tensorflow as tf
from PIL import Image


class Data:

    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        (self.mnist_x_train, self.mnist_y_train), (self.mnist_x_test, self.mnist_y_test) = self.mnist.load_data()
        self.mnist_x_train, self.mnist_x_test = self.mnist_x_train / 255.0, self.mnist_x_test / 255.0

        IMAGE_SIZE = self.mnist_x_train.shape[-1]

        self.mnist_x_train = self.mnist_x_train.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))
        self.mnist_x_test = self.mnist_x_test.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))

        self.usps_features = []
        self.usps_labels = []
        self.cur_path = r'../datasets/USPSdata/Numerals'

    def pre_process(self):
        print('processing dataset . . .')

        for j in range(0, 10):
            cur_folder_path = self.cur_path + r'/' + str(j)
            imgs = os.listdir(cur_folder_path)

            for img in imgs:
                cur_img = cur_folder_path + r'/' + img

                if cur_img[-3:] == 'png':
                    img = Image.open(cur_img, 'r')
                    img = img.resize((28, 28))
                    imgdata = (255 - np.array(img.getdata())) / 255
                    self.usps_features.append(imgdata)
                    self.usps_labels.append(j)

        testing_data = np.concatenate((self.mnist_x_test, np.asarray(self.usps_features)), axis=0)
        testing_label = np.concatenate((self.mnist_y_test, np.asarray(self.usps_labels)), axis=0)

        print('data processing complete !')

        return self.mnist_x_train, self.mnist_y_train, self.mnist_x_test, self.mnist_y_test, \
               np.asarray(self.usps_features), np.asarray(self.usps_labels), testing_data, testing_label
