import numpy as np
import tensorflow as tf
import tflearn
import os

# TODO: change output layer to be 5 and 3 for adding no-op

def get_movement_model():
    # Network building
    net = tflearn.input_data(shape=[None, 10, 62720], name='net1_layer1')
    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net1_layer2')
    net = tflearn.dropout(net, 0.6, name='net1_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net1_layer4')
    net = tflearn.dropout(net, 0.6, name='net1_layer5')
    net = tflearn.fully_connected(net, 5, activation='softmax', name='net1_layer6')
    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
                             name='net1_layer7')
    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0)


def get_action_model():
    # Network building
    net = tflearn.input_data(shape=[None, 10, 62720], name='net2_layer1')
    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net2_layer2')
    net = tflearn.dropout(net, 0.6, name='net2_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net2_layer4')
    net = tflearn.dropout(net, 0.6, name='net2_layer5')
    net = tflearn.fully_connected(net, 3, activation='softmax', name='net2_layer6')
    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
                             name='net2_layer7')
    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0)


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


def get_list(n_samples=10000):
    list = []
    for i in range(0, n_samples):
        feature_vector = np.random.rand(128, 1)
        output_movement = np.zeros((4, 1))
        output_movement[np.random.randint(0, 4), 0] = 1
        output_action = np.zeros((2, 1))
        output_action[np.random.randint(0, 4), 0] = 1
        list.append([feature_vector, output_movement, output_action])
    return list

