from __future__ import print_function
from scipy.io import loadmat as load


train = load('train_32x32.mat')
test = load('test_32x32.mat')

# Let's verify that they are dict type
print(type(train), type(test))

# Get samples and labels
train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']

# Build Computational Graph
# Todo: tensorflow stuff
