import time, numpy as np
from directkeys import PressKey, ReleaseKey, Q, W, E, S, A, D


def superattack():
    PressKey(Q)
    time.sleep(0.05)
    ReleaseKey(Q)

def attack():
    PressKey(E)
    time.sleep(0.05)
    ReleaseKey(E)

def releaseAllKeys():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def front():
    # releaseAllKeys()
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(W)

def left():
    # releaseAllKeys()
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(A)

def right():
    # releaseAllKeys()
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(D)
    
def back():
    # releaseAllKeys()
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(S)

def countdown(t):
    for i in list(range(t))[::-1]:
        print(i+1)
        time.sleep(1)


def take_action(movement_index, action_index):
    if movement_index == 0:
        # print('left')
        left()
    elif movement_index == 1:
        # print('front')
        front()
    if movement_index == 2:
        # print('right')
        right()
    elif movement_index == 3:
        # print('back')
        back()
    time.sleep(0.2)
    if action_index == 0:
        print('attack')
        attack()
    elif action_index == 1:
        print('superattack')
        superattack()
    movement_map = {
        0: A,
        1: W,
        2: D,
        3: S,
        4: ''
    }

    action_map = {
        0: E,
        1: Q,
        2: ''
    }

    # action_code = action_map[action_index]
    # movement_code = movement_map[movement_index]
    # if movement_code == '':
    #     releaseAllKeys()
    
    # PressKey(movement_code)
    # PressKey(action_code)
    # time.sleep(0.02)
    # ReleaseKey(action_code)
    # time.sleep(0.5)
    # ReleaseKey(movement_map[movement_index])