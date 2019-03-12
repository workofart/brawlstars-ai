import numpy as np
from object_detection.grabscreen import grab_screen
import cv2
import time
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from getkeys import key_check, keys_to_action, keys_to_movement
from utilities import countdown
from mobilenet import mnet_feature

file_name = 'data/training_data_bounty_attack_mobilenet.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

def main():
    countdown(4)

    while(True):
        # 800x600 windowed mode
        screen = grab_screen(region=(0,30,1280,745))
        last_time = time.time()
        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (224,224))
        keys = key_check()
        output = keys_to_movement(keys)
        training_data.append([screen,output])
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)

def main_feature():
    model = mnet_feature()
    # countdown(4)
    print('started')
    while(True):
        # 800x600 windowed mode
        screen = grab_screen(region=(0,30,1280,745))
        last_time = time.time()
        screen = cv2.resize(screen, (224,224))

        # Feed the raw screen into MobileNet to get features
        # The features are in the size of (None, 7, 7, 320)
        features = model.predict([screen.reshape(1,224,224,3)])[0]

        keys = key_check()
        movement = keys_to_movement(keys)
        action = keys_to_action(keys)

        training_data.append([features,movement,action])
        
        if 'P' in keys:
            print('Quitting')
            break

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)

main_feature()