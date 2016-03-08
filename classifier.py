# Use this script to train a single layer perceptron neural network.
# Adjust 'steps to train for longer/shorter.
# Adjust 'batchSize' to change SGD batch size.
# Adjust 'accStep' to display/calculate accuracy every 'accStep' training steps.
# The output network is stored in network.pickle in bianry format.

import numpy as np
import tensorflow as tf
import pickle as pickle
import matplotlib.pyplot as plt

hidden_size = 64

def main():
	# Load the dataset
	dataset = pickle.load( open( 'alphadataset.pickle', "rb" ) )
	train = dataset['train']
	trainLabels = dataset['trainLabels']
	test = dataset['test']
	testLabels = dataset['testLabels']
	imsize = train.shape[1]


	# Network parameters
	x = tf.placeholder(tf.float32, [None,imsize])
	W = tf.Variable(tf.truncated_normal([imsize, hidden_size], stddev=0.1))
	b = tf.Variable(tf.zeros([hidden_size]))
	y1 = tf.nn.relu(tf.matmul(x, W) + b)

	W1 = tf.Variable(tf.truncated_normal([hidden_size,26], stddev=0.1))
	b1 = tf.Variable(tf.zeros([26]))


	y = tf.nn.softmax(tf.matmul(y1, W1)+b1)

	y_ = tf.placeholder(tf.float32, [None, 26])

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
	train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

	init = tf.initialize_all_variables()

	sess = tf.Session()

	sess.run(init)

	# Run SGD training
	trainLength = train.shape[0]
	batchSize = 512
	epochs = 80
	steps = epochs * (trainLength//batchSize)
	print("epochs:", epochs)
	accStep = steps//10
	acc = [] # store accuracy at steps for plotting

	i_index = 0;
	for i in range(steps):

		#shuffle
		if (i_index >= trainLength//batchSize):
			i_index = 0;
			randindices = np.arange(trainLength)
			np.random.shuffle(randindices)
			train = train[np.array(randindices)]
			trainLabels = trainLabels[np.array(randindices)]

		indices = np.arange(train.shape[0])
		start = i_index*batchSize
		end = start + batchSize

		indices = indices[start:end]

		batch_xs, batch_ys = (train[indices], trainLabels[indices])
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		if i%accStep == 0:
			pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
			acc.append(sess.run(accuracy, feed_dict={x:test, y_:testLabels}))
			print("Accuracy: ",acc[-1])

		i_index += 1

	# Save network info
	network = {}
	network['weights'] = np.array(sess.run(W))
	network['bias'] = np.array(sess.run(b))

	network['weights1'] = np.array(sess.run(W1))
	network['bias1'] = np.array(sess.run(b1))

	pickle.dump(network, open('network.pickle', 'wb'))

	sess.close()

	plt.plot(acc)
	plt.title('Training accuracy per step')
	plt.xlabel('steps')
	plt.ylabel('accuracy')
	plt.show()

if __name__ == "__main__":
    main()
