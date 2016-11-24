import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle

def main():
    # test an image here
    img = cv2.imread("school.jpg")
    img7 = img
    print(img.shape)
    img2 = cv2.imread("bus.jpg")
    img22= img2
    img3 = cv2.imread("no_standing.jpg")
    img4 = img3
    img5 = img4
    img6 = img5
    all_images = []
    all_images.append(img)
    all_images.append(img7)
    all_images.append(img2)
    all_images.append(img22)
    all_images.append(img3)
    all_images.append(img4)
    all_images.append(img5)
    all_images.append(img6)
    labels = ["school", "school", "bus", "bus", "no_standing", "no_standing", "no_standing", "no_standing"]

    balance(all_images, labels, 10)

    for item in all_images:
        plt.imshow(item)
        # plt.show()
        plt.imshow(jitter(item))
        # plt.show()


def balance(features, labels, total_size):
    # take in an unbalanced data set (training images, one-hot labels)
    # create a balanced data set by:
    # - downsampling the ones for which there are too many
    # - upsampling the ones for which we have too few

    uniques = set(labels)
    count_dict = {}
    for item in uniques:
        count_dict[item] = labels.count(item)

    print(count_dict)
    max_count = max(count_dict[item] for item in uniques)

    # resample set so that there are maximum_count examples of each
    new_features = []
    new_labels = []

    combo = zip(features, labels)

    for i in uniques:
        source_features = []
        source_labels = []
        for j in combo:
            if j[1] == i:
                source_features.append(j[0])
                source_features.append(j[1])
        resampled_features, resampled_labels = resample(source_features, source_labels, max_count)
        new_features.append(resampled_features)
        new_labels.append(resampled_labels)

    # now reshuffle everything
    final_features, final_labels = resample(new_features, new_labels, len(new_features))

    return final_features, final_labels


def jitter_all(images):
    # take in a list of images and jitter them all
    new_images = []
    for i in range(0, len(images)):
        new_images[i] = jitter(new_images)
    return


def resample(features, labels, n):
        # take in a list of features/labels pairs and randomly return n of them
        # automatically upsample if needed
        index = list(range(0, len(features) - 1))
        shuffle(index)
        print(index)

        rs_features = []
        rs_labels = []
        for i in range(0, n):
            print("i: {}".format(i))
            var = index[(i+1) % len(features) - 1]
            print(features[var])
            print(var)
            rs_features.append(features[var])
            rs_labels.append(labels[var])
        return rs_features, rs_labels


def jitter(img):
    # take in an image and return a jittered image
    h,w,c = img.shape
    noise = np.random.randint(0, 50, (h,w))
    zitter = np.zeros_like(img)
    zitter[:,:,1] = noise

    noise_added = cv2.add(img, zitter)
    # combined = np.vstack((img[:int(h/2),:,:], noise_added[int(h/2):,:,:]))
    return noise_added


if __name__ == "__main__":
    main()