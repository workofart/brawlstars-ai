import numpy as np


raw = np.load('data/training_data_bounty_attack_mobilenet.npy')
converted_data = []
for data in raw:
    # data[0]
    if data[1] == [0,0,0,0]:
        data[1] = [0,0,0,0,1]
    else:
        data[1].append(0)
    if data[2] == [0,0]:
        data[2] = [0,0,1]
    else:
        data[2].append(0)
    
    converted_data.append([data[0], data[1], data[2]])

np.save('data/converted.npy',converted_data)