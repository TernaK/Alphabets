# Use this script to read the dataset from the AlphabetsDataset directory.
# It performs adaptive thresholding. Foreground is black!
# Use 'side' to set the image size desired.
# The output is stored in alphadataset.pickle in binary format.

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import sys
import pickle as pickle
import numpy as np
import cv2

side = 28
im_size = side**2
samples = 512*26

alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphaLabels = {}
label = np.zeros([1,26])
label[0,0] = 1.
for letter in alphabets:
	alphaLabels[letter] = np.array(label)
	label = np.roll(label, 1)

# Preprocessing applied to all training images
def preprocess(img):
	return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)

def readData():
	# Read in each image, threshold vectorize, save in array, save label (one-hot style)
	alldata = np.zeros([samples, im_size])
	allLabels = np.zeros([samples, 26])
	index = 0
	
	root = 'AlphabetDataset'
	for folder in alphabets:
		files = os.listdir(root+'/'+folder)
		for file in files:
			if(file.endswith('.DS_Store')):
				continue
			filename = root+'/'+folder+'/'+file
			img = cv2.imread(filename, 0)
			img = cv2.resize(img, (side, side))
			img = preprocess(img)
			alldata[index] = np.array(img).reshape([1,im_size])/255.
			allLabels[index] = alphaLabels[folder]
			index += 1

	return (alldata, allLabels)


def getDataset():
	# Separate data into training and test data. Set num_test appropriately.
	# Randomly shuffle the data first.
	allData, allLabels = readData()
	num_test = 2000
	num_train = samples-num_test

	dataset = {'train':np.zeros([num_train,im_size]), 'trainLabels':np.zeros([num_train,26]), 'test':np.zeros([num_test,im_size]), 'testLabels':np.zeros([num_test,26])}
	indices = np.arange(samples)
	np.random.shuffle(indices)

	for i in range(0, num_train):
		dataset['train'][i] = np.array(allData[indices[i]])
		dataset['trainLabels'][i] = np.array(allLabels[indices[i]])
		pass

	for i in range(0,num_test):
		dataset['test'][i] = np.array(allData[indices[i+num_train]])
		dataset['testLabels'][i] = np.array(allLabels[indices[i+num_train]])
		pass

	return dataset

dataset = getDataset()

for key in dataset.keys():
	print(key + ": ", dataset[key].shape)

# dump in a pickle file
pickle.dump(dataset, open('alphadataset.pickle', 'wb'))
