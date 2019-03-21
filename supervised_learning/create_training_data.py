import numpy as np
from object_detection.grabscreen import grab_screen
import cv2
import time
import os
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
from utilities.getkeys import key_check, keys_to_action, keys_to_movement
from utilities.utilities import countdown
# from net.mobilenet import mnet_feature

file_name = 'data/training_data_bounty_attack_raw_screen_200_200.npy'

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # config.log_device_placement = True

# sess = tf.Session(config=config)
# set_session(sess)

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
    # model = mnet_feature()
    # countdown(4)
    
    print('started')
    counter = 0
    temp_storage = []
    while(True):
        screen = grab_screen(region=(0,30,1280,745))
        last_time = time.time()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (200,200))

        # Feed the raw screen into MobileNet to get features
        # The features are in the size of the output layer Conv_1 (Conv2D) -> (None, 7, 7, 1280)
        # features = model.predict([screen.reshape(1,224,224,3)])[0]

        keys = key_check()
        movement = keys_to_movement(keys)
        action = keys_to_action(keys)

        # Store raw screen so preprocessing can be done later
        # training_data.append([screen,movement,action])
        temp_storage.append([screen,movement,action])
        
        if 'O' in keys:
            while True:
                print('waiting...')
                time.sleep(2)
                keys = key_check()
                if 'O' in keys:
                    print('resumed')
                    break
        if 'P' in keys:
            print('Quitting, and saving data')
            
            if os.path.isfile(file_name):
                print('File exists, loading previous data!')
                training_data = list(np.load(file_name))
            else:
                print('File does not exist, starting fresh!')
                training_data = []
            save_file = np.append(training_data, temp_storage, axis=0)
            print('Total frames:{}'.format(len(save_file)))
            np.save(file_name, save_file)
            print('done!')
            break

        if counter % 500 == 0:
            print(counter)
            # print(len(training_data))
            # np.save(file_name,training_data)
        counter += 1

main_feature()