import numpy as np
import cv2
import tensorflow as tf
from random import shuffle
import time

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
grey_train = {}
grey_train['features'] = []
for i in range(len(X_train)):
    grey_train['features'].append(rgb_to_gray(X_train[i]))
train['features'] = grey_train['features']
# print("Train features type: {}".format(type(train['features'])))


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


# The actual CNN itself

# Parameters
learning_rate = 0.005
iterations = 20000
train_batch_size = 100
test_batch_size = 50
display_step = 10

# Shuffle training set
index = list(range(0, len(train['features']) - 1))
shuffle(index)

shuffled_features = []
shuffled_labels = []
for i in index:
    shuffled_features.append(train['features'][i])
    shuffled_labels.append(one_hot_train_labels[i])

# Create batches
total_batch = int(n_train / train_batch_size) + 1
feature_batches = np.array_split(shuffled_features, total_batch)
label_batches = np.array_split(shuffled_labels, total_batch)

n_input = image_shape[0] * image_shape[1]  # traffic sign input (img shape: 32*32)

# Conv layer configurations
# -------------------------
# Conv layer 1
filter_size1 = 5    # 5x5
num_filters1 = 16   # use 16 of these filters

# Conv layer 2
filter_size2 = 5    # 5x5
num_filters2 = 36   # use 36 of these filters

# Fully connected layer
fc_size = 128

# ------------------

# Define some helper functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.0, shape=[length]))


def new_conv_layer(input,               # Previous layer
                   num_input_channels,  # Num channels from previous layer
                   filter_size,         # Width x height of each filter, if 5x5 enter 5 here
                   num_filters,         # Number of filters
                   use_pooling=True):    # Use 2x2 max pooling or not
    # shape determined by Tensorflow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create some new weights for the shape above and initialize them randomly
    weights = new_weights(shape=shape)

    # Create one bias for each filter
    biases = new_biases(length=num_filters)

    # Create Tensorflow convolution operation
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],  # first and last stride must always be 1
                         padding='SAME')        # padding: what to do at edge of image

    # Add biases to the reuslts of convolution:
    layer += biases

    # Use pooling if indicated:
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1,2,2,1],
                              padding='SAME')

    # Then use a RELU to introduce some non-linearity
    layer = tf.nn.relu(layer)

    # ReLU is normally executed before pooling
    # but relu(max_pool(x)) == max_pool(relu(x))
    # So would rather run ReLU on a smaller piece (1x1 as opposed to 2x2)

    # return both layer and filter weights for later use when running the session
    return layer, weights


# Helper function to flatten a layer, i.e. when feeding form a conv layer into a fully connected
def flatten_layer(layer):
    # Get shape of input
    input_shape = layer.get_shape()

    # format of shape should be [num_images, img_height, img_width, num_channels]
    # total # of features is therefore img_height * img_width * num_channels; grab this
    num_features = input_shape[1:4].num_elements()

    # flatten to 2D, leaving the first dimension open
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


# Helper function to make a fully connected layer
def new_fc_layer(input,             # previous layer
                 num_inputs,        # number of inputs from previous layer
                 num_outputs,       # of outputs
                 use_relu=True):    # Use a ReLU or not

    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Layer is matrix mult of inputs by weights, plus bias
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Get some image dimensions
img_size = 32
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1        # Grayscale so only one channel
num_classes = n_classes

# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
# convert from one-hot to the class number
y_true_cls = tf.argmax(y_true, dimension=1)
# for dropout
keep_prob = tf.placeholder("float")

# Make conv layer 1, takes in image
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels = num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

# Make conv layer 2, takes in output of layer 1
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels = num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)



# Make the flat layer, takes in output of conv 2
layer_flat, num_features = flatten_layer(layer_conv2)

# Make fully connected layer 1, takes in output of flat layer, output is # of neurons
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu = True)

# stick a dropout layer between these two
dropout = tf.nn.dropout(layer_fc1, keep_prob)

# Make fully connected layer 2, which takes in 128 things and outputs a vector of 10 (logits)
logits = new_fc_layer(input = layer_fc1,
                         num_inputs = fc_size,
                         num_outputs=num_classes,
                         use_relu=False)    # Don't use ReLU on final layer; pass to a softmax


# pass logits into softmax, get predictions out in the form of probabilities
y_pred = tf.nn.softmax(logits)  # DON'T FEED THIS INTO tf.nn.softmax_cross_entropy_with_logits()

y_pred_cls = tf.argmax(y_pred, dimension=1)

# # Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# DEFINE SOME PERFORMANCE MEASURES
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# this creates a vector of trues and falses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# turn previous vector into a % value; this returns TRAINING accuracy

def print_test_accuracy(session):
    # Number of images in the test-set.
    num_test = len(test['features'])

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # The starting index for the next batch is denoted i.
    print("Calculating test accuracy...")
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = test['features'][i:j]

        # Get the associated labels.
        labels = one_hot_test_labels[i:j]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,
                     keep_prob: 1}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = test['labels']

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))


# # Initializing the variables
init = tf.initialize_all_variables()

# # Launch the graph
with tf.Session() as sess:
    start_time = time.time()
    sess.run(init)
    # Training cycle
    for i in range(0, iterations):
        # grab a batch
        batch_x = feature_batches[i % total_batch]
        batch_y = label_batches[i % total_batch] # hacky solution for making sure we always have a batch
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y_true: batch_y,
                                       keep_prob: 0.5})
        # Display logs per epoch step
        if i % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x,
                                          y_true: batch_y,
                                          keep_prob: 1})
            acc = sess.run(accuracy, feed_dict={x: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 1})
            print("Iteration {}:\tCost={:.5},\tTraining accuracy={:.1%}".format(i, c, acc))
    print("Optimization Finished!")
    print("--- %s seconds ---" % (time.time() - start_time))
    print_test_accuracy(sess)




# notes to self of stuff to do:
# 0) FILL IN PYTHON NOTEBOOK
# 1) getting validation set
# 2) upsampling to make categories balanced, + jitter
# *** use scikit learn train test split

# *** try jittering: rotation sheer and translation i guess. merge this w upsamling to get more balanced set
# take initial dataset and move the pixels around a bit but not to much so you can still recongnize the sign
# makes the network more robust against overfitting since you have more images
