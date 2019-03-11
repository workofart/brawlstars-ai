import numpy as np
from PIL import ImageGrab
import cv2
import time, random
from directkeys import PressKey,ReleaseKey, W, A, S, D, Q, E
from alexnet import alexnet
from getkeys import key_check

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'brawlstars-bounty-attack-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR, EPOCHS)
model.load(MODEL_NAME)

def superattack():
    PressKey(Q)
    time.sleep(0.01)
    ReleaseKey(Q)

def attack():
    PressKey(E)
    time.sleep(0.01)
    ReleaseKey(E)

def front():
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(W)

def left():
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(S)
    PressKey(A)

def right():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    PressKey(D)
    
def back():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(S)


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
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,60))
            action_probs = model.predict([screen.reshape(80,60,1)])[0]
            # To prevent very low probs from taking over
            # and going into infinite loop
            # print(max(action_probs))
            if max(action_probs) > 0.3:
                
                moves = list(np.around(action_probs))
                print(moves)
                
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
main()