import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 7
MODEL_NAME = 'brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)
LOAD_MODEL_NAME = 'brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',15)

model = alexnet(WIDTH, HEIGHT, LR, EPOCHS)
# model.load(LOAD_MODEL_NAME)
train_data = np.load('data/training_data_bounty_attack.npy')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)