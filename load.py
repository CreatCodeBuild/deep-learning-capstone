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
print(train_labels[2000])
def reformat(dataset, labels):
	# Because each element in labels is a length 1 array,
	# we need to get rid of it
	# num_images = dataset.shape[3]
	labels = np.array([x[0] for x in labels])	# slow code, whatever
	# newArray = np.full(
	# 	shape=(num_images, image_size, image_size, num_channels),
	# 	fill_value=0.0,
	# 	dtype=np.float32
	# )
	# for i in range(num_images):
	# 	for x in range(image_size):
	# 		for y in range(image_size):
	# 			for z in range(num_channels):
	# 				newArray[i][x][y][z] = dataset[x][y][z][i]

	# dataset = dataset \
	# 	.reshape((-1, image_size, image_size, num_channels)) \
	# 	.astype(np.float32)
	dataset = np.transpose(dataset, (3, 0, 1, 2)).astype(np.float32)
	# labels = (np.arange(num_labels) == labels[:,None].astype(np.float32)
	one_hot_labels = []
	for num in labels:
		one_hot = [0.0] * 10
		if num == 10:
			one_hot[0] = 1.0
		else:
			one_hot[num] = 1.0
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	return dataset, labels


train_dataset, train_labels = reformat(train_samples, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_samples, test_labels)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	plt.imshow(train_dataset[3000])
	print(train_labels[3000])
	plt.show()
	plt.imshow(train_dataset[2000])
	print(train_labels[2000])
	plt.show()
	# exploration
	# d = {}
	# for l in train_labels:
	# 	if l[0] in d:
	# 		d[l[0]] += 1
	# 	else:
	# 		d[l[0]] = 1
	# for k, v in d.items():
	# 	print(k, v)
