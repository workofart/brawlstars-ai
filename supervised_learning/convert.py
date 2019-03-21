import tables
import numpy as np
import h5py

filename = 'temp.h5'
old_name = 'data/training_data_bounty_attack_raw_screen_200_200.npy'
# ROW_SIZE = 100
# NUM_COLUMNS = 200
training_data = np.load(old_name)
hf = h5py.File('data.h5', 'w')
screen = []
movement = []
action = []
for idx in range(len(training_data)):
    screen.append(training_data[idx][0].tolist())
    movement.append(training_data[idx][1])
    action.append(training_data[idx][2])
hf.create_dataset('screen', data=screen)
hf.create_dataset('movement', data=movement)
hf.create_dataset('action', data=action)
hf.close()