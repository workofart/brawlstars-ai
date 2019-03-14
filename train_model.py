import numpy as np
import os
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model
from net.alexnet import alexnet
from net.mobilenet import mnet
from net.lstm import reshape_for_lstm, get_action_model, get_movement_model


WIDTH = 80
HEIGHT = 60
LR = 3e-5
EPOCHS = 500

train_data = np.load('data/training_data_bounty_attack_mobilenet.npy')

def preprocess(data, length, WIDTH, HEIGHT):
    train = data[:-length]
    test = data[-length:]

    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    return X, Y, test_x, test_y

def train_alexnet():
    MODEL_NAME = 'models/brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)
    LOAD_MODEL_NAME = 'models/brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',20)
    model = alexnet(WIDTH, HEIGHT, LR, EPOCHS)
    model.load(LOAD_MODEL_NAME)
    
    X, Y, test_x, test_y = preprocess(train_data, 2500, WIDTH, HEIGHT)

    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

def train_mnet():
    MODEL_NAME = 'models/brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'mobilenetv2',EPOCHS)
    if os.path.isfile(MODEL_NAME):
        print('Model exists, loading previous model!')
        net = load_model(MODEL_NAME)
    else:
        print('Model does not exist, starting fresh!')
        net = mnet()
        net.compile(Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Construct training/testing data
    X = np.array([i[0] for i in train_data])
    Y = np.array([i[1] for i in train_data])

    net.fit(x=X, y=Y, batch_size=12, epochs=EPOCHS, verbose=2, validation_split=0.1)
    net.save(MODEL_NAME)

def train_lstm():
    data = list(train_data)
    print('Loaded supervised learning data: ' + str(np.shape(data)))

    # preprocess training data
    X, Y_movement, Y_action = reshape_for_lstm(data)

    with tf.Graph().as_default():
        model_movement = get_movement_model()
        model_movement.fit(X, Y_movement, n_epoch=5, validation_set=0.1,batch_size=8)
        model_movement.save('models/movement/movement_model')

    with tf.Graph().as_default():
        model_action = get_action_model()
        model_action.fit(X, Y_action, n_epoch=5, validation_set=0.1,batch_size=8)
        model_action.save('models/action/action_model')


train_lstm()