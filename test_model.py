import numpy as np
from PIL import ImageGrab
import cv2
import time, random
from directkeys import PressKey,ReleaseKey, W, A, S, D, Q, E
from alexnet import alexnet
from mobilenet import mnet, mnet_feature
from lstm import get_action_model, get_movement_model
from utilities import countdown, take_action
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from getkeys import key_check
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

WIDTH = 80
HEIGHT = 60
LR = 3e-5
EPOCHS = 500
# MODEL_NAME = 'brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)
MODEL_NAME = 'models/brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'mobilenetv2', EPOCHS)
ACTION_MODEL = 'models/action/action_model'
MOVEMENT_MODEL = 'models/movement/movement_model'

# model = alexnet(WIDTH, HEIGHT, LR, EPOCHS)
# model = mnet()
# model = load_model(MODEL_NAME)

def create_input_window(model, steps_of_history=10, size_of_frame=62720):
    input_window = np.zeros(shape=(steps_of_history, size_of_frame))

    for i in range(0, steps_of_history):
        screen =  np.array(ImageGrab.grab(bbox=(0,30,1280,745)))
        screen = cv2.resize(screen, (224,224))
        features = model.predict([screen.reshape(1,224,224,3)])[0]
        input_window[i, :] = np.array(features).reshape(-1, 62720)
    return input_window

# [A,W,D,S]
def main():
    last_time = time.time()
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            screen =  np.array(ImageGrab.grab(bbox=(0,30,1280,745)))
            # print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (224,224))
            action_probs = model.predict([screen.reshape(1,224,224,3)])[0]
            # To prevent very low probs from taking over
            # and going into infinite loop
            # print(max(action_probs))
            if max(action_probs) > 0.3:
                
                moves = list(np.around(action_probs))
                # print(moves)
                
                if moves == [1,0,0,0,0,0]:
                    print('left')
                    left()
                elif moves == [0,1,0,0,0,0]:
                    print('front')
                    front()
                elif moves == [0,0,1,0,0,0]:
                    print('right')
                    right()
                elif moves == [0,0,0,1,0,0]:
                    print('back')
                    back()
                elif moves == [0,0,0,0,1,0]:
                    print('attack')
                    attack()
                elif moves == [0,0,0,0,0,1]:
                    print('superattack')
                    superattack()
   
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)


def main_feature():
    g1 = tf.Graph()
    g2 = tf.Graph()
    model = mnet_feature()
    with g1.as_default():
        action_model = get_action_model()
        action_model.load(ACTION_MODEL)
    
    with g2.as_default():
        movement_model = get_movement_model()
        movement_model.load(MOVEMENT_MODEL)
    paused = False
    print('Starting to play')
    countdown(5)
    # Start by capturing a moving screen now
    input_window = create_input_window(model)
    current_time = time.time()
    
    while(True):
        
        if not paused:
            screen =  np.array(ImageGrab.grab(bbox=(0,30,1280,745)))
            last_time = time.time()
            print(last_time - current_time)
            current_time = last_time
            screen = cv2.resize(screen, (224,224))
            features = model.predict([screen.reshape(1,224,224,3)])[0]

            # Append the newest screen feature to the moving window of features
            input_window[:-1, :] = input_window[1:, :]
            input_window[-1, :] = np.array(features).reshape(-1, 62720)

            with g1.as_default():
                actions = np.round(np.array(action_model.predict(input_window.reshape(-1, 10, 62720))))
            
            with g2.as_default():
                movements = np.round(np.array(movement_model.predict(input_window.reshape(-1, 10, 62720))))

            selected_movement = np.argmax(movements)
            selected_action = np.argmax(actions)
            # if (actions < 0.3).all():
            #     selected_action = 2 # no action
            # if (movements < 0.3).all():
            #     selected_movement = 4 # no movement
            # print('Selected Action: {0}'.format(selected_action))
            # print('Selected Movement: {0}'.format(selected_movement))

            take_action(selected_movement, selected_action)
            keys = key_check()

            if 'P' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    cv2.destroyAllWindows()
                    time.sleep(1)
            elif 'O' in keys:
                print('Quitting!')
                cv2.destroyAllWindows()
                break


main_feature()