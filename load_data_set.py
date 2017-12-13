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


#读取图片数据并于标签绑定
def read_path(images, labels, path_name, label):
    for dir_item in os.listdir(path_name):

        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        image = cv2.imread(full_path)
        image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)     ##把所有图片改成64*64大小的
        images.append(image)
        labels.append(label)

def loaddata(parent_dir):
    images = []
    labels = []
    read_path(images, labels, parent_dir+"xiang", 0)
    read_path(images, labels, parent_dir+"nong", 1)
    read_path(images, labels, parent_dir+"cheng", 2)
    read_path(images, labels, parent_dir+"others3", 3)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

if __name__ == '__main__':
    images, labels = loaddata("C:/Users/xiang/Pictures/face/")
