from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np

# Load data
train = load('data/train_32x32.mat')
test = load('data/test_32x32.mat')

# Get samples and labels
train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']

# global configuration / hyper parameters
num_labels = 10
image_size = 32
num_channels = 1

def reformat(dataset, labels):
	# Because each element in labels is a length 1 array,
	# we need to get rid of it
	# num_images = dataset.shape[3]
	labels = np.array([x[0] for x in labels])	# slow code, whatever
	dataset = np.transpose(dataset, (3, 0, 1, 2)).astype(np.float32)
	one_hot_labels = []
	for num in labels:
		one_hot = [0.0] * 10
		if num == 10:
			one_hot[0] = 1.0
		else:
			one_hot[num] = 1.0
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	# linearly normalize the image value from 0 - 255 to -1.0 to 1.0
	return dataset, labels

# to Gray-scale
def normalize(dataset):
	a = np.add.reduce(dataset, keepdims=True, axis=3)
	a = a/3.0
	# print(a.shape)
	return a/128.0 - 1.0

_train_dataset, _train_labels = reformat(train_samples, train_labels)
_test_dataset, _test_labels = reformat(test_samples, test_labels)

_train_dataset = normalize(_train_dataset)
_test_dataset = normalize(_test_dataset)

def inspect(i, normalized):
	if normalized:
		plt.imshow(train_dataset[i])
	else:
		plt.imshow(_train_dataset[i])
	plt.show()
	print(train_labels[i])

def distribution(labels, name):
	count = {}
	for label in labels:
		if (0 if label[0] == 10 else label[0]) in count:
			count[0 if label[0] == 10 else label[0]] += 1
		else:
			count[0 if label[0] == 10 else label[0]] = 1
	x = []
	y = []
	for k, v in count.items():
		print(k, v)
		x.append(k)
		y.append(v)
	# draw x, y
	objects = x
	y_pos = np.arange(len(objects))
	performance = y
	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title(name + ' Label Distribution')
	plt.show()


### todo: Try Gray-scale it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if __name__ == '__main__':
	# exploration
	# get data distribution
	# distribution(train_labels, 'Train')
	# distribution(test_labels, 'Test')
	inspect(1000, _train_dataset)
