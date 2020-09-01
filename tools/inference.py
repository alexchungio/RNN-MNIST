#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/30 下午4:37
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def show_mnist(image):
    """

    :param image:
    :return:
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    # https://matplotlib.org/examples/color/colormaps_reference.html
    ax1.imshow(image)
    ax1.set_title("default")
    ax2.imshow(image, cmap=plt.get_cmap('gray'))
    ax2.set_title("gray")
    ax3.imshow(image, cmap=plt.get_cmap('gray_r'))
    ax2.set_title("gray_r")
    ax4.imshow(image, cmap=plt.get_cmap('viridis'))
    ax2.set_title("viridis")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    train_images, train_labels = mnist.train.images, mnist.train.labels
    test_images, test_labels = mnist.test.images, mnist.test.labels

    image = mnist.train.images[0].reshape(-1, 28)
    # convert image format
    # image = np.reshape(image, (28, 28, 1))
    # new_image = np.concatenate([image, image, image], axis=-1)

    show_mnist(image)