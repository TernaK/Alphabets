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
	f.write('float Network_weights[' + str(im_size*26) + '] = {\n')
	for col in range(26):
		for row in range(im_size):
			f.write('\t' + str(network['weights'][row][col]) + ',\n')
	f.write('};\n\n')


	f.write('float Network_biases[26] = {\n')
	for i in range(26):
		f.write('\t' + str(network['bias'][i]) + ',\n')
	f.write('};')