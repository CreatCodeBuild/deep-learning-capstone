# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import tensorflow as tf
from six.moves import range
import numpy as np
import csv
### My module
import load



# define our computational graph
# hyper parameters
num_hidden = 64
batch_size = 128
patch_size = 5	# filter size
conv1_depth = 32
conv2_depth = 32	# just for semantic clarity
conv3_depth = 16	# just for semantic clarity
conv4_depth = 16	# just for semantic clarity
last_conv_depth = conv4_depth
pooling_stride = 2
drop_out_rate = 0.9

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels # R G B
num_steps = 10001


def run_session():
	def accuracy(predictions, labels):
	  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
	          / predictions.shape[0])

	def run_dataset(samples, labels, record_csv):
		'''
		@return: average loss, average accuracy
		'''
		with open(record_csv, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=['iteration', 'loss', 'accuracy'])
			writer.writeheader()
			total_loss = 0
			total_accu = 0
			for step in range(num_steps):
				offset = (step * batch_size) % (labels.shape[0] - batch_size)
				batch_data = samples[offset:(offset + batch_size), :, :, :]
				batch_labels = labels[offset:(offset + batch_size), :]
				feed_dict = {
					tf_train_dataset : batch_data,
					tf_train_labels : batch_labels
				}
				_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
				total_loss += l
				accu = accuracy(predictions, batch_labels)
				total_accu += accu
				writer.writerow({'iteration': step, 'loss': l, 'accuracy': accu})
				if (step % 50 == 0):
					print('Minibatch loss at step %d: %f' % (step, l))
					print('Minibatch accuracy: %.1f%%' % accu)
					# print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
			return total_loss/num_steps, total_accu/num_steps

	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		# ###
		print('Start Training')
		average_loss, average_accuracy = run_dataset(train_dataset, train_labels, 'record/train.csv')
		print('Average Loss:', average_loss)
		print('Average Accuracy:', average_accuracy)
		###
		print('Start Testing')
		average_loss, average_accuracy = run_dataset(test_dataset, test_labels, 'record/test.csv')
		print('Average Loss:', average_loss)
		print('Average Accuracy:', average_accuracy)


### Start
train_dataset, train_labels = load.train_dataset, load.train_labels
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = load.test_dataset,  load.test_labels
print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


graph = tf.Graph()
with graph.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	# tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset  = tf.constant(test_dataset)

	# Variables.
	# conv1 layer 1
	# "layer1_weights" is a terrible naming, better to name it "conv1_filter"
	conv1_filter = tf.Variable(
		tf.truncated_normal([3, 3, num_channels, conv1_depth], stddev=0.1))
	conv1_biases = tf.Variable(tf.zeros([conv1_depth]))

	# conv layer 2
	conv2_filter = tf.Variable(
		tf.truncated_normal([3, 3, conv1_depth, conv2_depth], stddev=0.1))
	conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_depth]))

	# conv layer 3
	conv3_filter = tf.Variable(
		tf.truncated_normal([3, 3, conv2_depth, conv3_depth], stddev=0.1))
	conv3_biases = tf.Variable(tf.constant(0.1, shape=[conv3_depth]))

	conv4_filter = tf.Variable(
		tf.truncated_normal([3, 3, conv3_depth, conv4_depth], stddev=0.1))
	conv4_biases = tf.Variable(tf.constant(0.1, shape=[conv4_depth]))


	# layer 3, fully connected
	down_scale = pooling_stride ** 2	# because we do 2 times pooling of stride 2
	layer3_weights = tf.Variable(
		tf.truncated_normal(
			[image_size // down_scale * image_size // down_scale * last_conv_depth, num_hidden],
			stddev=0.1))
	layer3_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden]))

	# layer 4
	layer4_weights = tf.Variable(
		tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	layer4_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

	# Model.
	def model(data, isTrain=False):
		# conv layer 1
		conv1 = tf.nn.conv2d(data, conv1_filter, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv1 + conv1_biases)

		# conv layer 2
		conv2 = tf.nn.conv2d(hidden, conv2_filter, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv2 + conv2_biases)
		hidden = tf.nn.max_pool(
			hidden,
			[1,pooling_stride,pooling_stride,1],
			[1,pooling_stride,pooling_stride,1],
			padding='SAME')

		# conv layer 3
		conv3 = tf.nn.conv2d(hidden, conv3_filter, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv3 + conv3_biases)

		conv4 = tf.nn.conv2d(hidden, conv4_filter, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv4 + conv4_biases)
		hidden = tf.nn.max_pool(
			hidden,
			[1,pooling_stride,pooling_stride,1],
			[1,pooling_stride,pooling_stride,1],
			padding='SAME')


		# fully connected layer 1
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

		# fully connected layer 2
		# Add a 75% dropout during training only. Dropout also scales
		# activations such that no rescaling is needed at evaluation time.
		if isTrain:
			hidden = tf.nn.dropout(hidden, drop_out_rate, seed=4926)

		return tf.matmul(hidden, layer4_weights) + layer4_biases

	# Training computation.
	logits = model(tf_train_dataset, True)
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	# L2 regularization for the fully connected parameters
	regularization = tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + \
	                 tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases)
	loss += 5e-4 * regularization

	# learning rate decay
	# todo: momentum?
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(
		0.0013,
		global_step * batch_size,
		100,
		0.99,
		staircase=True
	)

	# Optimizer.
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	# optimizer = tf.train \
	# 	.MomentumOptimizer(learning_rate, 0.2) \
	# 	.minimize(loss, global_step=global_step)
	optimizer = tf.train \
		.AdamOptimizer(learning_rate) \
		.minimize(loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	# valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

if __name__ == '__main__':
	run_session()
