import numpy as np
import tensorflow as tf
import pickle as pickle
import matplotlib.pyplot as plt
from math import sqrt

dataset = pickle.load( open( 'alphadataset.pickle', "rb" ) )
train = dataset['train']
trainLabels = dataset['trainLabels']
test = dataset['test']
testLabels = dataset['testLabels']
imsize = train.shape[1]
side = int(sqrt(imsize))

def weight_variable(shape):
	weights = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(weights)

def bias_variable(shape):
	biases = tf.constant(0.1, shape=shape)
	return tf.Variable(biases)

def conv2d(x, W):
 	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
 	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, imsize])
x_image = tf.reshape(x, [-1,side,side,1])

W_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([4*4*16, 512])
b_conv3 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*16])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_conv3) + b_conv3)
#h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([512, 26])
b_fc1 = bias_variable([26])

y_conv = tf.nn.relu(tf.matmul(h_fc1, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#W_fc2 = weight_variable([1024, 26])
#b_fc2 = bias_variable([26])

#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 26])

#cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


sess.run(tf.initialize_all_variables())

stepSize = 200
steps = 7000
acc = []
for i in range(steps):
	indices = np.arange(train.shape[0])
	np.random.shuffle(indices)
	indices = indices[200:stepSize+200]
	batch_xs, batch_ys = (train[indices], trainLabels[indices])

	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	if i % 100 == 0:
		correct_predictions = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
		print("Test accuracy: %g" % accuracy.eval(feed_dict={x: test, y_: testLabels}))


sess.close()
