# Use this script to generate variables for a C/C++ program which will 
# use the weights and biases for the network in network.pickle.

import numpy as np
import pickle as pickle
import cv2
import matplotlib.pyplot as plt
from readdataset import side

alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
im_size = side**2
network = pickle.load(open('network.pickle', 'rb'))


with open('netparams.h', 'w') as f:
	f.write('//SIDE: ' + str(side) + '\n')
	f.write('float Network_weights1[' + str(im_size*32) + '] = {\n')
	for col in range(32):
		for row in range(im_size):
			f.write('\t' + str(network['weights'][row][col]) + ',\n')
	f.write('};\n\n')


	f.write('float Network_biases1[32] = {\n')
	for i in range(32):
		f.write('\t' + str(network['bias'][i]) + ',\n')
	f.write('};\n\n')

	f.write('float Network_weights2[' + str(32*26) + '] = {\n')
	for col in range(26):
		for row in range(32):
			f.write('\t' + str(network['weights1'][row][col]) + ',\n')
	f.write('};\n\n')


	f.write('float Network_biases2[26] = {\n')
	for i in range(26):
		f.write('\t' + str(network['bias1'][i]) + ',\n')
	f.write('};')