#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import os

from tensorflow.examples.tutorials.mnist import input_data

import deep_learning_models
import models
from BayesianDecision import DiscriminantAnalysis
from confusion_matrix import ConfusionMatrix
from dataset import Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable the warning, doesn't enable AVX/FMA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def print_and_plot(model, accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined):
    print('\n\n----------' + model + '----------')
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot(model + ' on MNIST dataset',
                          confusion_mnist, confusion_mnist.shape[0])

    print('\nOn USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_usps))
    confusion_matrix.plot(model + ' on USPS dataset',
                          confusion_usps, confusion_usps.shape[0])

    print('\nOn Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_combined))
    confusion_matrix.plot(model + ' on combined dataset',
                          confusion_combined, confusion_combined.shape[0])


def main():
    data = Data()
    logistic_regression = models.LogisticRegression()
    neural_network = models.NeuralNet()
    svm = models.SupportVectorMachine(C=1.0, kernel='rbf', gamma='scale')
    random_forest = models.RandomForest(n_estimators=100, max_depth=None, random_state=None)
    discriminant_analysis = DiscriminantAnalysis()
    vaecnn = deep_learning_models.VAEConvolutionalNeuralNet(
        input_data.read_data_sets("data", one_hot=True),
        (28, 28),
        (28, 28)
    )

    # Process dataset
    training_data_features, training_data_labels, mnist_test_data_features, mnist_test_data_labels, \
    usps_test_data_features, usps_test_data_labels, combined_test_data_features, combined_test_data_labels = \
        data.pre_process()

    # Discriminant Analysis
    IMAGE_SIZE = int(training_data_features.shape[-1] ** 0.5)
    discriminant_analysis.fit(training_data_features.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)), training_data_labels)
    accuracy_mnist, confusion_mnist = discriminant_analysis.predict(
        'MNIST dataset', mnist_test_data_features.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)), mnist_test_data_labels)
    accuracy_usps, confusion_usps = discriminant_analysis.predict(
        'USPS dataset', usps_test_data_features.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)), usps_test_data_labels)
    accuracy_combined, confusion_combined = discriminant_analysis.predict(
        'Combined dataset', combined_test_data_features.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)),
        combined_test_data_labels)
    print_and_plot('Bayesian Discriminant Analysis', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Logistic Regression
    logistic_regression.fit(training_data_features, training_data_labels, learning_rate=0.01, epochs=500)
    accuracy_mnist, confusion_mnist = logistic_regression.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = logistic_regression.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = logistic_regression.predict(
        combined_test_data_features, combined_test_data_labels)
    print_and_plot('Logistic Regression', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Neural Network
    neural_network.fit(training_data_features, training_data_labels, epochs=10)
    accuracy_mnist, confusion_mnist = neural_network.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = neural_network.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = neural_network.predict(
        combined_test_data_features, combined_test_data_labels)
    print_and_plot('Neural Network', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Support Vector Machine
    svm.fit(training_data_features, training_data_labels)
    accuracy_mnist, confusion_mnist = svm.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = svm.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = svm.predict(combined_test_data_features, combined_test_data_labels)
    print_and_plot('SVM', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Random Forest
    random_forest.fit(training_data_features, training_data_labels)
    accuracy_mnist, confusion_mnist = random_forest.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = random_forest.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = random_forest.predict(
        combined_test_data_features, combined_test_data_labels)
    print_and_plot('Random Forest', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Restricted Boltzmann Machine
    num_hidden_nodes_list = [20, 100, 500]
    for num_hidden_nodes in num_hidden_nodes_list:
        rbm = deep_learning_models.RBM(
            images=input_data.read_data_sets("data", one_hot=True),
            n_components=num_hidden_nodes,
            learning_rate=0.02,
            batch_size=100,
            n_iter=1000,
            random_state=0
        )
        rbm.fit()
        rbm.gibbs_sampling(1000)
        rbm.generate_images(num_hidden_nodes)

    # Variational Auto Encoders
    code_unit_list = [2, 8, 16]
    for code_unit in code_unit_list:
        vae = deep_learning_models.VAE(input_data.read_data_sets("data", one_hot=True), code_unit)
        vae.generate_images(epochs=20)

    # Variational Auto Encoders with Convolutional Neural Networks
    vaecnn.encode()
    vaecnn.decode()
    vaecnn.compile_()
    vaecnn.train(epochs=10, batch_size=100)


if __name__ == '__main__':
    confusion_matrix = ConfusionMatrix()
    main()
