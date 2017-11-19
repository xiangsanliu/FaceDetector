import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64

def resize_image(image, herght = IMAGE_SIZE, width = IMAGE_SIZE):
    return cv2.resize(image, width, width)

images = []
labels = []

def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):
            read_path(full_path)
        else :
            image = cv2.imread(full_path)
            image = cv2.resize(image, IMAGE_SIZE, IMAGE_SIZE)
            images.append(image)
            labels.append(path_name)
    return images, labels

def loaddata(path_name):
    images, labels = read_path(path_name)

    images = np.array(images)
    print(images.shape)

    labels = np.array([0 if label.endswith('xiang') else 1 for label in labels])
    return images, labels

images, labels = loaddata("")