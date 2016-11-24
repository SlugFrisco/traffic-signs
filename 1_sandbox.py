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

# HELPER STUFF-----------------------------------------
# Get signnames.csv and put it into a dict
with open('signnames.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    labels_dict = {rows[0]:rows[1] for rows in reader}
# 'dtype = object' allows the np.ndarray to take in strings
new_labels = np.ndarray(len(train['labels']), dtype=object)


# Takes in a numerical label and returns the right text string
def get_text(num_label):
    return labels_dict[str(num_label)]


# Plot an example of each type
def plot_all(features, labels):
    displayed = []
    index = 0
    plot_num = 1

    plt.figure(figsize=(15, 10))
    for item in labels:
        if item not in displayed:
            plt.subplot(8, 6, plot_num)
            plt.imshow(features[index], cmap='gray')
            plt.title(get_text(labels[index]), fontsize=8)
            plt.axis('off')
            displayed.append(item)
            plot_num += 1
        index += 1
    plt.show()

plot_all(train['features'], train['labels'])

