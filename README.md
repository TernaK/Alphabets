# Alphabets
This is a single layer neural network (26 neurons) which is used to classify images of alphabets.

Training
========
Training was done using Tensorflow.

Dataset
=======

The dataset consists of 512 grayscale images each of scrabble tiles for each letter of the alphabet.
Small variations were created by moving the tiles around and rotating them slightly.

Files
=====
* dataset.cpp uses OpenCV to capture the data. 
* readdataset.py uses OpenCV to read the images and preprocess them by adaptive thresholding.
* classifier.py uses Tensorflow to train the network.
* networktester.py models the network using the weights and biases learned and can be used to test imput images.
