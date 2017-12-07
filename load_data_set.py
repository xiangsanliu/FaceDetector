import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    h, w, _ = image.shape

    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        righ = dw - left
    else:
        pass

    BLACK = [0, 0, 0]

    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))


def read_path(images, labels, path_name, label):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        image = cv2.imread(full_path)
        image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
        images.append(image)
        labels.append(label)


def loaddata(parent_dir):
    images = []
    labels = []
    read_path(images, labels, parent_dir+"xiang", 0)
    read_path(images, labels, parent_dir+"nong", 1)
    read_path(images, labels, parent_dir+"unknow", 2)

    images = np.array(images)
    labels = np.array(labels)

    # labels = np.array([0 if label.endswith('xiang') else 1 for label in labels])
    # labels = np.array([1 if label.endswith('wang') else label for label in labels])
    # labels = np.array([2 if label.endswith('cheng') else label for label in labels])


    return images, labels


if __name__ == '__main__':
    images, labels = loaddata("C:/Users/xiang/Documents/face/")
