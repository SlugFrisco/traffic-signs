import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    # test an image here
    img = cv2.imread("school.jpg")
    plt.imshow(img)
    plt.show()
    plt.imshow(jitter(img))
    plt.show()

def balance_by_jittering(features, labels, total_size):
    # take in an unbalanced data set (training images, one-hot labels)
    # create a balanced data set by jittering
    # and downsampling the ones for which there are too many
    # upsample the ones for which we have too few
    return True


def resample(all_examples, n):
        # take in an array of examples and randomly return n of them
    return True


def jitter(img):
    # take in an image and return a jittered image
    h,w,c = img.shape
    noise = np.random.randint(0, 50, (h,w))
    zitter = np.zeros_like(img)
    zitter[:,:,1] = noise

    noise_added = cv2.add(img, zitter)
    combined = np.vstack((img[:int(h/2),:,:], noise_added[int(h/2):,:,:]))
    return combined


if __name__ == "__main__":
    main()