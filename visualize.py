# To visualize the intermediate layers of the CNN

import numpy as np 
import matplotlib as mp
import cv2
import matplotlib.pyplot as plt
import math

# import tensorflow as tf
# import tensorflow.contrib.slim as slim

# tf.reset_default_graph()

# x = tf.placeholder(tf.float32, [100, 100, 3],name="x-in")

# x_image = tf.reshape(x,[-1,100,100,1])
# hidden_1 = slim.conv2d(x_image,5,[5,5])
# pool_1 = slim.max_pool2d(hidden_1,[2,2])
# hidden_2 = slim.conv2d(pool_1,5,[5,5])
# pool_2 = slim.max_pool2d(hidden_2,[2,2])
# hidden_3 = slim.conv2d(pool_2,20,[5,5])

# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

from net.mobilenet import mnet_feature

net = mnet_feature()



def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[100, 100, 3],order='F')})
    plotNNFilter(units)


def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(100,100))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.show()


imageToUse = cv2.imread('object_detection/img/bounty/gameplay_3.png')[0:705, 0:1280]
img_small = cv2.resize(imageToUse, (100, 100))
getActivations(hidden_3,img_small)

# cv2.imshow('img', img_small)
# cv2.waitKey()