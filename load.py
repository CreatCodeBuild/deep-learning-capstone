from __future__ import print_function
from scipy.io import loadmat as load
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
num_channels = 3


def reformat(dataset, labels):
	# Because each element in labels is a length 1 array,
	# we need to get rid of it
	labels = np.array([x[0] for x in labels])	# slow code, whatever
	dataset = dataset \
		.reshape((-1, image_size, image_size, num_channels)) \
		.astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels


train_dataset, train_labels = reformat(train_samples, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_samples, test_labels)

if __name__ == '__main__':
	# exploration
	d = {}
	for l in train_labels:
		if l[0] in d:
			d[l[0]] += 1
		else:
			d[l[0]] = 1
	for k, v in d.items():
		print(k, v)
