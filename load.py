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
	return dataset/128.0 - 1.0, labels


train_dataset, train_labels = reformat(train_samples, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_samples, test_labels)

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	plt.imshow(train_dataset[4000])
	print(train_labels[4000])
	plt.show()

	for x in train_labels:
		if sum(x) != 1.0:
			print(x)
	print(sum([sum(x) for x in train_labels]))

	# exploration
	# d = {}
	# for l in train_labels:
	# 	if l[0] in d:
	# 		d[l[0]] += 1
	# 	else:
	# 		d[l[0]] = 1
	# for k, v in d.items():
	# 	print(k, v)
