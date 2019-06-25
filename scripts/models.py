#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from scripts.confusion_matrix import ConfusionMatrix

confusion_matrix = ConfusionMatrix()


class LogisticRegression:

    def __init__(self):
        self.w = np.array(0)
        self.yhat = []

    # Initializing the weights 'normally'
    def init_weights(self, shape):
        """
        :param shape: dimensionality of the weight matrix
        :return: void
        """
        self.w = np.random.randn(shape, 10)

    @staticmethod
    def find_confusion_matrix(y, yhat):
        """
        :param y: actual label
        :param yhat: predicted label
        :return: confusion matrix
        """
        return confusion_matrix.calculate(y, yhat)

    # Calculate Softmax
    @staticmethod
    def softmax(z):
        """
        :param z:
        :return: softmax
        """
        return (np.exp(z) / np.sum(np.exp(z), axis=0)).T

    def predict(self, x, y):
        """
        :param x: testing feature-vector
        :param y: testing label
        :return: prediction accuracy and confusion matrix
        """
        y_ = np_utils.to_categorical(np.array(y), 10)
        z = np.dot(self.w.T, x.T)
        self.yhat = self.softmax(z)

        num_of_wrong_predictions = 0
        num_of_right_predictions = 0

        for i, j in zip(y_, self.yhat):
            if np.argmax(i) == np.argmax(j):
                num_of_right_predictions = num_of_right_predictions + 1
            else:
                num_of_wrong_predictions = num_of_wrong_predictions + 1

        self.yhat = np.argmax(self.yhat, axis=1)
        accuracy = num_of_right_predictions / (num_of_right_predictions + num_of_wrong_predictions)
        confusion = self.find_confusion_matrix(y, self.yhat)

        return accuracy, confusion

    def fit(self, x, y, learning_rate, epochs):
        """
        :param x: training feature-vector
        :param y: training label
        :param learning_rate: learning rate
        :param epochs: number of epochs
        :return: void
        """
        print('\nTraining using Logistic Regression')
        self.init_weights(x.shape[1])
        y = np_utils.to_categorical(np.array(y), 10)
        for _ in tqdm(range(epochs), total=epochs):
            z = np.dot(self.w.T, x.T)
            yhat = self.softmax(z)
            gradient = np.dot(x.T, yhat - y)
            gradient /= len(x.T)
            gradient *= learning_rate
            self.w -= gradient


class NeuralNet:

    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.yhat = []

    @staticmethod
    def find_confusion_matrix(y, yhat):
        """
        :param y: actual label
        :param yhat: predicted label
        :return: confusion matrix
        """
        return confusion_matrix.calculate(y, yhat)

    def predict(self, x, y):
        """
        :param x: testing feature-vector
        :param y: testing label
        :return: prediction accuracy and confusion matrix
        """
        self.yhat = np.argmax(self.model.predict(x), axis=1)
        accuracy = metrics.accuracy_score(y, self.yhat)
        confusion = self.find_confusion_matrix(y, self.yhat)

        return accuracy, confusion

    def fit(self, x, y, epochs):
        """
        :param x: training feature-vector
        :param y: training label
        :param epochs: number of epochs
        :return: void
        """
        print('\nTraining using Neural Network')
        self.model.fit(x, y, epochs=epochs)


class SupportVectorMachine:

    def __init__(self, C, kernel, gamma):
        # Classify as SVM
        self.classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)

        self.yhat = []

    @staticmethod
    def find_confusion_matrix(y, yhat):
        """
        :param y: actual label
        :param yhat: predicted label
        :return: confusion matrix
        """
        return confusion_matrix.calculate(y, yhat)

    def predict(self, x, y):
        """
        :param x: testing feature-vector
        :param y: testing label
        :return: prediction accuracy and confusion matrix
        """
        self.yhat = self.classifier.predict(x)
        accuracy = metrics.accuracy_score(y, self.yhat)
        confusion = self.find_confusion_matrix(y, self.yhat)

        return accuracy, confusion

    def fit(self, x, y):
        """
        :param x: training feature-vector
        :param y: training label
        :return: void
        """
        print('\nTraining using SVM')
        self.classifier.fit(x, y)


class RandomForest:

    def __init__(self, n_estimators, max_depth, random_state):
        # Classify as Random Forest Model
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

        self.yhat = []

    @staticmethod
    def find_confusion_matrix(y, yhat):
        """
        :param y: actual label
        :param yhat: predicted label
        :return: confusion matrix
        """
        return confusion_matrix.calculate(y, yhat)

    def fit(self, x, y):
        """
        :param x: training feature-vector
        :param y: training label
        :return: void
        """
        print('\nTraining using Random Forest')
        self.classifier.fit(x, y)

    def predict(self, x, y):
        """
        :param x: testing feature-vector
        :param y: testing label
        :return: prediction accuracy and confusion matrix
        """
        self.yhat = self.classifier.predict(x)
        accuracy = metrics.accuracy_score(y, self.yhat)
        confusion = self.find_confusion_matrix(y, self.yhat)

        return accuracy, confusion
