import numpy as np
from alexnet import alexnet
from mobilenet import mnet
from keras.optimizers import Adam

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'models/brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)
LOAD_MODEL_NAME = 'models/brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',20)

train_data = np.load('data/training_data_bounty_attack_mobilenet.npy')

def train_alexnet():
    model = alexnet(WIDTH, HEIGHT, LR, EPOCHS)
    model.load(LOAD_MODEL_NAME)
    
    train = train_data[:-2500]
    test = train_data[-2500:]

    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

def train_mnet():
    net = mnet()
    net.compile(Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    # Construct training/testing data
    X = np.array([i[0] for i in train_data])
    Y = np.array([i[1] for i in train_data])

    net.fit(x=X, y=Y, epochs=EPOCHS, verbose=2, validation_split=0.1)

train_mnet()