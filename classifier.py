# Use this script to train a single layer perceptron neural network.
# Adjust 'steps to train for longer/shorter.
# Adjust 'batchSize' to change SGD batch size.
# Adjust 'accStep' to display/calculate accuracy every 'accStep' training steps.
# The output network is stored in network.pickle in bianry format.

import numpy as np
import tensorflow as tf
import pickle as pickle
import matplotlib.pyplot as plt

# Load the dataset
dataset = pickle.load( open( 'alphadataset.pickle', "rb" ) )
train = dataset['train']
trainLabels = dataset['trainLabels']
test = dataset['test']
testLabels = dataset['testLabels']
imsize = train.shape[1]

# Network parameters
x = tf.placeholder(tf.float32, [None,imsize])
W = tf.Variable(tf.zeros([imsize,26]))
b = tf.Variable(tf.zeros([26]))
y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder(tf.float32, [None, 26])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

# Run SGD training
batchSize = 256
steps = 10000
accStep = 1000
acc = [] # store accuracy at steps for plotting
for i in range(steps):
	indices = np.arange(train.shape[0])
	np.random.shuffle(indices)
	indices = indices[200:batchSize+200]

	batch_xs, batch_ys = (train[indices], trainLabels[indices])
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	if i%accStep == 0:
		pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
		acc.append(sess.run(accuracy, feed_dict={x:test, y_:testLabels}))
		print("Accuracy: ",acc[-1])

# Save network info
network = {}
network['weights'] = np.array(sess.run(W))
network['bias'] = np.array(sess.run(b))

pickle.dump(network, open('network.pickle', 'wb'))

sess.close()

plt.plot(acc)
plt.title('Training accuracy per step')
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.show()
