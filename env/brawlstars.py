import numpy as np
from mobilenet import mnet_feature
from PIL import ImageGrab
import cv2
import time

class Brawlstars():

    def __init__(self):
        self.reset()
        self.feature_model = mnet_feature()
        self.observation_space = []
        self.action_space = [0,1,2] # attack, super, no-op
        self.movement_space = [0,1,2,3,4] # left, front, right, back, no-op

    def _getReward(self):
        return

    def act(self):
        return

    def step(self, action):
        state = self._getObservation()
        done = self._isDone()
        reward = self._getReward()
        self.time_step += 1
        return state, reward, done

    def reset(self):
        self.time_step = 0
        return self._getObservation()

    # TODO: check if the game is over through OCR
    def _isDone(self):
        return

    def _getObservation(self):
        screen =  np.array(ImageGrab.grab(bbox=(0,30,1280,745)))
        last_time = time.time()
        # print(last_time - current_time)
        current_time = last_time
        screen = cv2.resize(screen, (224,224))
        features = self.feature_model.predict([screen.reshape(1,224,224,3)])[0]
        return features
    