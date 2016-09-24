# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import tensorflow as tf
from six.moves import range
import numpy as np
import csv
from sklearn.cross_validation import KFold
### My module
import load


### Start
train_dataset, train_labels = load._train_dataset, load._train_labels
test_dataset, test_labels = load._test_dataset,  load._test_labels
print('Training set', train_dataset.shape, train_labels.shape)
print('    Test set', test_dataset.shape, test_labels.shape)

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels # R G B

class Net():
	def __init__(self,
		num_hidden, batch_size, patch_size, conv1_depth, conv2_depth,
		pooling_stride, drop_out_rate, num_steps, optimizer,
		base_learning_rate, decay_rate,
		train_csv, test_csv):
		# hyper parameters
		self.num_hidden = num_hidden
		self.batch_size = batch_size
		self.patch_size = patch_size	# filter size
		self.conv1_depth = conv1_depth
		self.conv2_depth = conv2_depth
		self.last_conv_depth = conv2_depth
		self.pooling_stride = pooling_stride
		self.drop_out_rate = drop_out_rate
		self.num_steps = num_steps
		self.optimizer = optimizer # adam, momentum, gradient
		self.base_learning_rate = base_learning_rate
		self.decay_rate = decay_rate
		self.train_csv = train_csv
		self.test_csv = test_csv
		self.saver = None
		self.graph = None


	# define our computational graph
	def define_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			# Input data.
			tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, image_size, image_size, num_channels))
			tf_train_labels  = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
			# tf_valid_dataset = tf.constant(valid_dataset)
			tf_test_dataset  = tf.constant(test_dataset)

			# Variables.
			# conv1 layer 1
			# "layer1_weights" is a terrible naming, better to name it "conv1_filter"
			conv1_filter = tf.Variable(
				tf.truncated_normal([self.patch_size, self.patch_size, num_channels, self.conv1_depth], stddev=0.1))
			conv1_biases = tf.Variable(tf.zeros([self.conv1_depth]))

			# conv layer 2
			conv2_filter = tf.Variable(
				tf.truncated_normal([self.patch_size, self.patch_size, self.conv1_depth, self.conv2_depth], stddev=0.1))
			conv2_biases = tf.Variable(tf.constant(0.1, shape=[self.conv2_depth]))

			# layer 3, fully connected
			down_scale = self.pooling_stride ** 2	# because we do 2 times pooling of stride 2
			fc1_weights = tf.Variable(
				tf.truncated_normal(
					[image_size // down_scale * image_size // down_scale * self.last_conv_depth, self.num_hidden], stddev=0.1))
			fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))

			# layer 4
			fc2_weights = tf.Variable(
				tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
			fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

			# Model.
			def model(data, isTrain=False):
				# conv layer 1
				conv1 = tf.nn.conv2d(data, conv1_filter, [1, 1, 1, 1], padding='SAME')
				hidden = tf.nn.relu(conv1 + conv1_biases)
				hidden = tf.nn.max_pool(
					hidden,
					[1,self.pooling_stride,self.pooling_stride,1],
					[1,self.pooling_stride,self.pooling_stride,1],
					padding='SAME')

				# conv layer 2
				conv2 = tf.nn.conv2d(hidden, conv2_filter, [1, 1, 1, 1], padding='SAME')
				hidden = tf.nn.relu(conv2 + conv2_biases)
				hidden = tf.nn.max_pool(
					hidden,
					[1,self.pooling_stride,self.pooling_stride,1],
					[1,self.pooling_stride,self.pooling_stride,1],
					padding='SAME')

				# fully connected layer 1
				shape = hidden.get_shape().as_list()
				reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
				hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

				# fully connected layer 2
				if isTrain:
					hidden = tf.nn.dropout(hidden, self.drop_out_rate, seed=4926)
				return tf.matmul(hidden, fc2_weights) + fc2_biases

			# Training computation.
			logits = model(tf_train_dataset, True)
			loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

			# L2 regularization for the fully connected parameters
			regularization = tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + \
			                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases)
			loss += 5e-4 * regularization

			# learning rate decay
			# todo: momentum?
			global_step = tf.Variable(0)
			lr = self.base_learning_rate
			dr = self.decay_rate
			print(lr, dr, type(lr))
			learning_rate = tf.train.exponential_decay(
				# 0.0013,
				lr,
				global_step * self.batch_size,
				100,
				# 0.99,
				dr,
				staircase=True
			)

			# Optimizer.
			if self.optimizer == 'gradient':
				optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
			elif self.optimizer == 'momentum':
				optimizer = tf.train \
					.MomentumOptimizer(learning_rate, 0.5) \
					.minimize(loss, global_step=global_step)
			elif self.optimizer == 'adam':
				optimizer = tf.train \
					.AdamOptimizer(learning_rate) \
					.minimize(loss)
			else:
				raise Error('Wrong Optimizer')

			# Predictions for the training, validation, and test data.
			train_prediction = tf.nn.softmax(logits)
			test_prediction = tf.nn.softmax(model(tf_test_dataset))
			self.saver = tf.train.Saver(tf.all_variables())
		# print(tf_train_dataset, type(tf_train_dataset))
		# print(tf_train_labels)
		self.graph = graph
		return train_prediction, test_prediction, optimizer, loss, tf_train_dataset, tf_train_labels

	def run_session(self):
		train_prediction, test_prediction, optimizer, loss, tf_train_dataset, tf_train_labels = self.define_graph()
		def accuracy(predictions, labels):
			# print(type(predictions), type(labels))
		 	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

		def run_dataset(samples, labels, record_csv):
			'''
			@return: average loss, average accuracy
			'''
			with open(record_csv, 'w') as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=['iteration', 'loss', 'accuracy'])
				writer.writeheader()
				total_loss = 0
				total_accu = 0
				for step in range(self.num_steps):
					offset = (step * self.batch_size) % (labels.shape[0] - self.batch_size)
					batch_data = samples[offset:(offset + self.batch_size), :, :, :]
					batch_labels = labels[offset:(offset + self.batch_size), :]
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
				return total_loss/self.num_steps, total_accu/self.num_steps

		with tf.Session(graph=self.graph) as session:
			tf.initialize_all_variables().run()
			###
			print('Start Training')
			average_loss, average_accuracy = run_dataset(train_dataset, train_labels, self.train_csv)
			print('Average Loss:', average_loss)
			print('Average Accuracy:', average_accuracy)

			###
			### todo: Memory Explosion, TensorFlow issue
			###
			# print('Start Cross Validation')
			# kFold = KFold(len(train_dataset), n_folds=10, shuffle=True, random_state=1234)
			# for train_index, test_index in kFold:
			# 	kf_train_samples, kf_test_samples = train_dataset[train_index], train_dataset[test_index]
			# 	kf_train_labels, kf_test_labels = train_labels[train_index], train_labels[test_index]

			###
			### todo: Memory Explosion, TensorFlow issue
			###
			print('Start Testing')
			# average_loss, average_accuracy = run_dataset(test_dataset, test_labels, self.test_csv)
			save_path = self.saver.save(session, "model/model.ckpt")
			print("Model saved in file: %s" % save_path)
			# print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
			# print('Average Loss:', average_loss)
			# print('Average Accuracy:', average_accuracy)

	def test(self):
		# print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
		# print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
		if self.saver is None:
			graph, train_prediction, test_prediction, optimizer, loss, tf_train_dataset, tf_train_labels \
			= self.define_graph()
		with tf.Session(graph=graph) as session:
		  # Restore variables from disk.
		  self.saver.restore(session, "model/model.ckpt")
		  print("Model restored.")
		  print(type(tf.all_variables()))
		  # Do some work with the model


if __name__ == '__main__':
	# net1 = Net(
	# 	num_hidden=128,
	# 	batch_size=128,
	# 	patch_size=5,
	# 	conv1_depth=32,
	# 	conv2_depth=32,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
		# base_learning_rate=0.0013,
		# decay_rate=0.99,
		# optimizer='adam',
	# 	train_csv='record/train3.csv', test_csv='record/test3.csv'
	# )
	# net1.run_session()

	# net2 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=5,
	# 	conv1_depth=32,
	# 	conv2_depth=32,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
		# base_learning_rate=0.0013,
		# decay_rate=0.99,
		# optimizer='adam',
	# 	train_csv='record/train4.csv', test_csv='record/test4.csv'
	# )
	# net2.run_session()

	# net3 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=5,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
		# base_learning_rate=0.0013,
		# decay_rate=0.99,
		# optimizer='adam',
	# 	train_csv='record/train5.csv', test_csv='record/test5.csv'
	# )
	# net3.run_session()

	# net4 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=5,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.5,
	# 	num_steps=5001,
		# base_learning_rate=0.0013,
		# decay_rate=0.99,
		# optimizer='adam',
	# 	train_csv='record/train6.csv', test_csv='record/test6.csv'
	# )
	# net4.run_session()

	# net5 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=7,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
	# 	optimizer='adam',
	# 	base_learning_rate=0.0013,
	# 	decay_rate=0.99,
	# 	train_csv='record/train7.csv', test_csv='record/test7.csv'
	# )
	# net5.run_session()

	# net6 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=7,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
	# 	optimizer='adam',
	# 	base_learning_rate=0.005,
	# 	decay_rate=0.99,
	# 	train_csv='record/train8.csv', test_csv='record/test8.csv'
	# )
	# net6.run_session()

	# net7 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=7,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
	# 	optimizer='adam',
	# 	base_learning_rate=0.0005,
	# 	decay_rate=0.99,
	# 	train_csv='record/train9.csv', test_csv='record/test9.csv'
	# )
	# net7.run_session()

	# net8 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=7,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
	# 	optimizer='adam',
	# 	base_learning_rate=0.0013,
	# 	decay_rate=0.9,
	# 	train_csv='record/train10.csv', test_csv='record/test10.csv'
	# )
	# net8.run_session()

	# net9 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=7,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
	# 	optimizer='gradient',
	# 	base_learning_rate=0.0013,
	# 	decay_rate=0.99,
	# 	train_csv='record/train11.csv', test_csv='record/test11.csv'
	# )
	# net9.run_session()

	# net10 = Net(
	# 	num_hidden=64,
	# 	batch_size=64,
	# 	patch_size=7,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.9,
	# 	num_steps=5001,
	# 	optimizer='momentum',
	# 	base_learning_rate=0.0013,
	# 	decay_rate=0.99,
	# 	train_csv='record/train12.csv', test_csv='record/test12.csv'
	# )
	# net10.run_session()

	# netBenchmark = Net(
	# 	num_hidden=64,
	# 	batch_size=128,
	# 	patch_size=5,
	# 	conv1_depth=16,
	# 	conv2_depth=16,
	# 	pooling_stride=2,
	# 	drop_out_rate=0.5,
	# 	num_steps=5001,
	# 	optimizer='gradient',
	# 	base_learning_rate=0.05,
	# 	decay_rate=0.95,
	# 	train_csv='record/train_bench.csv', test_csv='record/test_bench.csv'
	# )
	# # netBenchmark.run_session()

	netDebug = Net(
		num_hidden=128,
		batch_size=128,
		patch_size=7,
		conv1_depth=32,
		conv2_depth=32,
		pooling_stride=2,
		drop_out_rate=0.9,
		num_steps=10001,
		optimizer='adam',
		base_learning_rate=0.0013,
		decay_rate=0.99,
		train_csv='record/train_debug.csv', test_csv='record/test_debug.csv'
	)
	# netDebug.run_session()
	netDebug.test()
	# netDebug.define_graph()
