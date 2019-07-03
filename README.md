# Image Classification and Reconstruction with MNIST and USPS Datasets

In this project we implement different machine learning models to classify and reconstruct the images in the MNIST dataset.


## Image Classification

We train our models using the MNIST training dataset and test the performance on the MNIST test dataset and the USPS dataset.

We train the following classifiers:
1. Bayesian Discriminant Function
2. Logistic Regression
3. SVM (Support Vector Machine) Package
4. Random Forest Package
5. Multilayer perceptron Neural Network

### Observations

1. Bayesian Discriminant Function, achieving an accuracy of 84%

| ![Bayesian Discriminant Analysis on MNIST dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Bayesian%20Discriminant%20Analysis%20on%20MNIST%20dataset.png)  | ![Bayesian Discriminant Analysis on USPS dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Bayesian%20Discriminant%20Analysis%20on%20USPS%20dataset.png)  | ![Bayesian Discriminant Analysis on Combined dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Bayesian%20Discriminant%20Analysis%20on%20combined%20dataset.png)  |
|---|---|---|


2. Logistic Regression, achieving an accuracy of 86%

| ![Logistic Regression on MNIST dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Logistic%20Regression%20on%20MNIST%20dataset.png)  | ![Logistic Regression on USPS dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Logistic%20Regression%20on%20USPS%20dataset.png)  | ![Logistic Regression on Combined datasets](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Logistic%20Regression%20on%20combined%20dataset.png)  |
|---|---|---|


3. SVM (Support Vector Machine) Package, achieving an accuracy of 97%

| ![SVM on MNIST dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/SVM%20on%20MNIST%20dataset.png)  | ![SVM on USPS dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/SVM%20on%20USPS%20dataset.png)  | ![SVM on Combined datasets](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/SVM%20on%20combined%20dataset.png)  |
|---|---|---|


4. Random Forest Package, achieving an accuracy of 97%

| ![Random Forest on MNIST dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Random%20Forest%20on%20MNIST%20dataset.png)  | ![Random Forest on USPS dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Random%20Forest%20on%20USPS%20dataset.png)  | ![Random Forest on Combined datasets](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Random%20Forest%20on%20combined%20dataset.png)  |
|---|---|---|


5. Multilayer perceptron Neural Network, achieving an accuracy of 98%

| ![Neural Network on MNIST dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Neural%20Network%20on%20MNIST%20dataset.png)  | ![Neural Network on USPS dataset](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Neural%20Network%20on%20USPS%20dataset.png)  | ![Neural Network on Combined datasets](https://github.com/siddheswarc/MNIST/raw/master/confusion_matrices/Neural%20Network%20on%20combined%20dataset.png)  |
|---|---|---|


## Image Reconstruction

We reconstruct the images in the MNIST dataset using the following models:
1. Restricted Boltzmann Machine (RBM)
2. Variational Auto Encoders (VAE)
3. Variational Auto Encoders with Convolution Neural Networks (VAE with CNNs)

### Observations

1. Restricted Boltzmann Machine (RBM)
![RBM reconstructed image with 500 hidden nodes](http://drive.google.com/uc?export=view&id=1dJ5mK9ffTuuD0J0yiMHFBLJftrZN89a7)

2. Variational Auto Encoders (VAE)
![VAE 16 code units - 20 Epochs](http://drive.google.com/uc?export=view&id=1FIm7I2pJDL3bl71MZFGFvORgcWq2A5-P)

3. Variational Auto Encoders with Convolution Neural Networks (VAE with CNNs)
![VAE CNN Reconstruction after 10 epochs](http://drive.google.com/uc?export=view&id=1wFhFKBGHcsSBKz-z0fYoJVKnb9vopkLN)
