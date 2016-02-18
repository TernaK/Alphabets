# Use this script to test the network which has been stored in network.pickle.
# Imput images will be read in grayscale and resized to 'side-by-'side' 
# (See readdataset.py).

import numpy as np
import pickle as pickle
import cv2
import matplotlib.pyplot as plt
from readdataset import side, preprocess

alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
im_size = side**2
network = pickle.load(open('network.pickle', 'rb'))

#tests = ['T/100.jpg', 'U/400.jpg', 'V/25.jpg', 'W/2.jpg', 'X/2.jpg', 'Y/100.jpg', 'Z/400.jpg', 'A/25.jpg',]

tests = ['_TEST/testg.jpg']
for test in tests:
	# Read, preprocess, vectorize
	img = cv2.imread('AlphabetDataset/'+test, 0)
	img = cv2.resize(img, (side, side))
	img = preprocess(img) 
	img_arr = np.array(img).reshape((1, im_size))/255.

	# Matrix multiply: z = x*W + b
	# Softmax: exp(z)/sum(exp(z))
	pred = np.dot(img_arr, network['weights']) + np.array(network['bias'])
	pred = np.exp(pred)
	pred = pred/np.sum(pred, 1)
	print('Predicted letter:',alphabets[np.argmax(pred)])
