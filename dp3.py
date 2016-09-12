import tensorflow as tf
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import SGDClassifier
import time

t0 = time.time()
#==================EXTRACTING AND ANALYSING DATA FROM .mat FILES===============
"""
X and Y Components of Training and Testing Data
"""
a = scipy.io.loadmat('data/train_32x32.mat')
train_data = a['X']
train_labels = a['y']
b = scipy.io.loadmat('data/test_32x32.mat')
test_data = b['X']
test_labels = b['y']
shape_train = train_data.shape
shape_test = test_data.shape
"""
Plotting Class Labels against their respective frequencies in a Bar Graph
"""
#temp_labels = train_labels.reshape(73257).tolist()
#temp_labels = dict(Counter(temp_labels))
#plt.bar(range(len(temp_labels)), temp_labels.values(), align='center', label='Training Labels')
#plt.xticks(range(len(temp_labels)), temp_labels.keys())
#temp_labels = test_labels.reshape(26032).tolist()
#temp_labels = dict(Counter(temp_labels))
#plt.bar(range(len(temp_labels)), temp_labels.values(), align='center', color='red', label='Testing Labels')
#plt.legend()
#plt.xlabel('Class Labels')
#plt.ylabel('Frequency')
#plt.title('Frequency Distribution of Class Labels')
#plt.show()
#============================================================================


print shape_train[3], "Images with", shape_train[0], "x", shape_train[0], "RGB grid"


#==================NORMALISATION AND PREPROCESSING=============================================


train_data = train_data.astype('float32') / 128.0 - 1
test_data = test_data.astype('float32') / 128.0 - 1


"""
Converting Labels to One Hot Encoding and Image Matrix to favourable dimensions
"""
def reformat(data, Y):
    xtrain = []
    trainLen = data.shape[3]
    for x in xrange(trainLen):
        xtrain.append(data[:,:,:,x])
    xtrain = np.asarray(xtrain)
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        elif el==1:
            temp[1]=1
        elif el==2:
            temp[2]=1
        elif el==3:
            temp[3]=1
        elif el==4:
            temp[4]=1
        elif el==5:
            temp[5]=1
        elif el==6:
            temp[6]=1
        elif el==7:
            temp[7]=1
        elif el==8:
            temp[8]=1
        elif el==9:
            temp[9]=1
        Ytr.append(temp)
    return xtrain, np.asarray(Ytr)

train_data, train_labels = reformat(train_data, train_labels)
test_data, test_labels = reformat(test_data, test_labels)


#============================================================================

#==================BUILDING THE CNN==========================================
"""
Various Hyperparameters required for training the CNN.
"""
image_size = 32
width = 32
height = 32
channels = 3

n_labels = 10
patch = 5
depth = 16
hidden = 128
dropout = 0.9375

batch = 128
learning_rate = 0.001

"""
Constructing the placeholders and variables in the TensorFlow Graph
"""
#Training Dataset
tf_train_dataset = tf.placeholder(tf.float32, shape=(None, width, height, channels))
#Training Labels
tf_train_labels = tf.placeholder(tf.float32, shape=(None, n_labels))
#Testing Dataset
tf_test_dataset = tf.constant(test_data)

#   Layer 1: (5, 5, 3, 16)
layer1_weights = tf.Variable(tf.truncated_normal([patch, patch, channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

#   Layer 2: (5, 5, 16, 16)
layer2_weights = tf.Variable(tf.truncated_normal([patch, patch, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

#   Layer 3: (1024, 128)
layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, hidden], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[hidden]))

#   Layer 4: (128, 10)                                                          1200 + 6400 + 131072 + 1280
layer4_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

dropout = tf.placeholder(tf.float32)

def model(data):
    #   Convolution 1 and RELU   [batch, in_height, in_width, in_channels]
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    #   Max Pool
    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #   Convolution 2 and RELU
    conv2 = tf.nn.conv2d(hidden2, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv2 + layer2_biases)
    #   Max Pool
    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    shape = hidden4.get_shape().as_list()

    reshape = tf.reshape(hidden4, [-1, shape[1] * shape[2] * shape[3]])
    hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    #   Dropout
    dropout_layer = tf.nn.dropout(hidden5, 0.93)

    return tf.matmul(dropout_layer, layer4_weights) + layer4_biases

logits = model(tf_train_dataset)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(model(tf_test_dataset))
#============================================================================

#==================TRAINING AND TESTING THE MODEL============================
"""
Accuracy function defined similar to the one taught in the Udacity Deep Learning Course
Returns percentage of correct predictions by verifying with Labels
"""
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#   Number of iterations
num_steps = 60000

saver = tf.train.Saver()

with tf.Session() as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average = 0
    for step in range(num_steps):
        #   Constucting the batch from the data set
        offset = (step * batch) % (train_labels.shape[0] - batch)
        batch_data = train_data[offset:(offset + batch), :, :, :]
        batch_labels = train_labels[offset:(offset + batch), :]
        #   Dictionary to be fed to TensorFlow Session
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, dropout: 0.93}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print "Average Accuracy : ", (average / num_steps)
    print "END OF TRAINING"
    average = 0

    for step in range(num_steps):
        #   Constucting the batch from the data set
        offset = (step * batch) % (test_labels.shape[0] - batch)
        batch_data = test_data[offset:(offset + batch), :, :, :]
        batch_labels = test_labels[offset:(offset + batch), :]
        #   Dictionary to be fed to TensorFlow Session
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, dropout: 0.93}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print "Average Accuracy : ", (average / num_steps)
    print "END OF TESTING"
    saver.save(session, 'my-model')
    #feed_dict = {tf_train_dataset : train_data[1, :, :, :], tf_train_labels : train_labesl[1], dropout: 0.93}
    #predictions = session.run(train_prediction, feed_dict=feed_dict)
    #print layer1_weights.eval()

    #batch_x = train_data[1, :, :, :]
    #predictions = sess.run(y_hat, feed_dict={x: batch_x})

    #classification = session.run(train_labels[1], feed_dict={tf_train_dataset : batch_x})
print time.time()-t0
#============================================================================
#batch_x = train_data[1, :, :, :]
#predictions = sess.run(y_hat, feed_dict={x: batch_x})

#classification = session.run(train_labels[1], feed_dict={tf_train_dataset : batch_x})
