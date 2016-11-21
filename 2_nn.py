import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from random import randint, shuffle
import csv

# Load pickled data
import pickle


# Helper function: convert an np.ndarray image from RGB to grayscale using OpenCV
def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

training_file = 'lab 2 data/train.p'
testing_file = 'lab 2 data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, Y_train = train['features'], train['labels']
X_test, Y_test = test['features'], test['labels']


# # Color to RGB, and normalize the numpy ndarray values to mean 0

# Keep originals for checking purposes
original_train = train.copy()
original_test = test.copy()

grey_train = {}
grey_train['features'] = []
for i in range(len(X_train)):
    grey_train['features'].append(rgb_to_gray(X_train[i]))
train['features'] = grey_train['features']

grey_test = {}
grey_test['features'] = []
for i in range(len(X_test)):
    grey_test['features'].append(rgb_to_gray(X_test[i]))
test['features'] = grey_test['features']


## Normalize: (value - 128) / 128
normalized_train = {}
normalized_train['features'] = []
for i in range(len(train['features'])):
    normalized_train['features'].append((train['features'][i] - 128) / 128)
train['features'] = normalized_train['features']

normalized_test = {}
normalized_test['features'] = []
for i in range(len(test['features'])):
    normalized_test['features'].append((test['features'][i] - 128) / 128)
test['features'] = normalized_test['features']


### To start off let's do a basic data summary.

n_train = len(train['features'])
n_test = len(test['features'])
image_shape = train['features'][0].shape
n_classes = len(set(train['labels']))


# # Show 10 example random greyscale and color images to make sure all is in order
# for i in range(0,9):
#     j = randint(0, n_train)
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_train['features'][j])
#     plt.subplot(1, 2, 2)
#     plt.imshow(train['features'][j], cmap='gray')
#     plt.show()


# Stolen helper function for one_hot encoding
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

one_hot_train_labels = dense_to_one_hot(train['labels'], n_classes)
one_hot_test_labels = dense_to_one_hot(test['labels'], n_classes)


# The actual Neural Network itself

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

# Shuffle training set
index = list(range(0, len(train['features']) - 1))
shuffle(index)

shuffled_features = []
shuffled_labels = []
shuffled_orig_labels = []
for i in index:
    shuffled_features.append(train['features'][i])
    shuffled_labels.append(one_hot_train_labels[i])

# Create batches
total_batch = int(n_train / batch_size) + 1
feature_batches = np.array_split(shuffled_features, total_batch)
label_batches = np.array_split(shuffled_labels, total_batch)

# HELPER STUFF-----------------------------------------
# Get signnames.csv and put it into a dict
with open('signnames.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    labels_dict = {rows[0]:rows[1] for rows in reader}
# 'dtype = object' allows the np.ndarray to take in strings
new_labels = np.ndarray(len(train['labels']), dtype=object)


# Define a helper function for later
# which takes in a numerical label and returns the right text string
def get_text(num_label):
    return labels_dict[str(num_label)]


# are labels right?
# print("One hot label for image 0 in shuffled: {}".format(shuffled_labels[0]))
# print("Normal label for image 0 in shuffled: {}".format(get_text(shuffled_orig_labels[0])))
# plt.imshow(shuffled_features[0])
# plt.show()


# # Plot an example of each type
# displayed = []
# index = 0
# plot_num = 1
#
# for item in shuffled_orig_labels:
#     if item not in displayed:
#         plt.subplot(8, 6, plot_num)
#         plt.imshow(shuffled_features[index], cmap='gray')
#         plt.title(get_text(item), fontsize=10)
#         plt.axis('off')
#         displayed.append(item)
#         plot_num += 1
#     index += 1
# plt.show()

# END HELPER STUFF ------------------------

n_input = image_shape[0] * image_shape[1]  # traffic sign input (img shape: 32*32)

n_hidden_layer = 256 # layer number of features.
# how many? http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# somewhere between 32*32 and the number of output classes, 43

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.truncated_normal([n_input, n_hidden_layer], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_layer, n_classes], stddev=0.1))
}
biases = {
    'hidden_layer': tf.Variable(tf.constant(0.0, shape=[n_hidden_layer])),
    'out': tf.Variable(tf.constant(0.0, shape=[n_classes]))
}

# tf Graph input
x = tf.placeholder("float", [None, image_shape[0], image_shape[1]])
y = tf.placeholder("float", [None, n_classes])

# tensorflow.python.framework.errors.InvalidArgumentError: logits and labels must be same size: logits_size=[300,43] labels_size=[100,43]
x_flat = tf.reshape(x, [-1, n_input])

# # Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# # Output layer with linear activation
logits = tf.matmul(layer_1, weights['out']) + biases['out']

# # Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# # Initializing the variables
init = tf.initialize_all_variables()

# # Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(n_train/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = feature_batches[i]
            batch_y = label_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test['features'], y: one_hot_test_labels}))

# notes to self of stuff to do:
# upsampling to make categories balanced
