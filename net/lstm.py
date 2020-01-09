import numpy as np
import tensorflow as tf
import tflearn
import os

def get_movement_model(steps):
    # Network building
    net = tflearn.input_data(shape=[None, steps, 128], name='net1_layer1')
    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net1_layer2')
    net = tflearn.dropout(net, 0.8, name='net1_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net1_layer4')
    net = tflearn.dropout(net, 0.8, name='net1_layer5')
    net = tflearn.fully_connected(net, 5, activation='softmax', name='net1_layer6')
    net = tflearn.regression(net, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0001,
                             name='net1_layer7')
    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_dir='logs', tensorboard_verbose=0)


def get_action_model(steps):
    # Network building
    net = tflearn.input_data(shape=[None, steps, 128], name='net2_layer1')
    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net2_layer2')
    net = tflearn.dropout(net, 0.8, name='net2_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net2_layer4')
    net = tflearn.dropout(net, 0.8, name='net2_layer5')
    net = tflearn.fully_connected(net, 3, activation='softmax', name='net2_layer6')
    net = tflearn.regression(net, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=1e-5,
                             name='net2_layer7')
    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_dir='logs', tensorboard_verbose=0)


def reshape_for_lstm(data, steps_of_history=10):
    trainX = []
    trainY_movement = []
    trainY_action = []

    for i in range(0, len(data) - steps_of_history):
        window = data[i:i + steps_of_history]

        sampleX = []
        for row in window:
            sampleX.append(row[0])
        sampleY_movement = np.array(window[-1][1]).reshape(-1)
        sampleY_action = np.array(window[-1][2]).reshape(-1)

        trainX.append(np.array(sampleX).reshape(steps_of_history, -1))
        trainY_movement.append(sampleY_movement)
        trainY_action.append(sampleY_action)

    print(np.array(trainX).shape)
    print(np.array(trainY_movement).shape)
    print(np.array(trainY_action).shape)

    return trainX, list(trainY_movement), list(trainY_action)
