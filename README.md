# Alphabets
This is a two-layer neural network (768-input, 64 hidden neurons, 26 output neurons) which is used to classify images of alphabets.

Training
========
Training was done using Tensorflow.

Dataset
=======

The dataset consists of 4096 grayscale images each of scrabble tiles for each letter of the alphabet.
Small variations were created by moving the tiles around and rotating them slightly.

Files
=====
* dataset.cpp uses OpenCV to capture the data. 
* readdataset.py uses OpenCV to read the images and preprocess them by adaptive thresholding.
* classifier.py uses Tensorflow to train the network.
* networktester.py models the network using the weights and biases learned and can be used to test input images.
* generatevars.py exports a C/C++ header file with the network weights and biases.
