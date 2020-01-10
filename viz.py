from keras.models import Model
from keras import layers
from keras.applications import vgg16, MobileNetV2
from keras.backend import set_session
from keras.preprocessing.image import save_img
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import time

# To visualize the intermediary CNN layers for debugging purposes.

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

# Reset the graph
sess = tf.InteractiveSession(config=config)
set_session(sess)

def _draw_filters(filters, layer_name, n=None, output_dim=(412,412)):
        """Draw the best filters in a nxn grid.
        # Arguments
            filters: A List of generated images and their corresponding losses
                     for each processed filter.
            n: dimension of the grid.
               If none, the largest possible square will be used
        """
        if n is None:
            n = int(np.floor(np.sqrt(len(filters))))

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top n*n filters.
        # filters.sort(key=lambda x: x[1], reverse=True)
        # filters = filters[:n * n]

        # build a black picture with enough space for
        # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 3), dtype='uint8')

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img = filters[i * n + j]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[
                    width_margin: width_margin + output_dim[0],
                    height_margin: height_margin + output_dim[1], :] = img

        # save the result to disk
        save_img('mobilenet_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)

def display_activation(activations, col_size, row_size, act_index): 
    """
    https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
    """
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='hsv')
            activation_index += 1
    plt.plot()
    plt.show()
        

start_time = time.time()
model = MobileNetV2()
# model.summary()

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs[1:-6])
imageToUse = cv2.imread('object_detection/img/bounty/gameplay_3.png')[0:705, 0:1280]
input_img_data = cv2.resize(imageToUse, (224, 224))

activations = activation_model.predict(input_img_data.reshape(1, 224, 224, 3))
print(time.time() - start_time)

_draw_filters(activations[1][0], 'test', output_dim=(112, 112))
# display_activation(activations, 4, 4, 128)