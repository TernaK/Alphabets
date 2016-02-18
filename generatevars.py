# Use this script to generate variables for a C/C++ program which will 
# use the weights and biases for the network in network.pickle.

import numpy as np
import pickle as pickle
import cv2
import matplotlib.pyplot as plt

alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
side = 28
im_size = side**2
network = pickle.load(open('network.pickle', 'rb'))