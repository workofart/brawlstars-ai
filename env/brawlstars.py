import numpy as np
from mobilenet import mnet_feature
from PIL import ImageGrab
from object_detection.processor import ScreenProcessor
from utilities.directkeys import PressKey, ReleaseKey, B
import cv2
import time

class Brawlstars():

    def __init__(self):
        self.reset()
        self.feature_model = mnet_feature()
        self.observation_space = []
        self.action_space = [0,1,2] # attack, super, no-op
        self.movement_space = [0,1,2,3,4] # left, front, right, back, no-op
        self.ScreenProcessor = ScreenProcessor(ImageGrab.grab(bbox=(0,30,1280,745)))
        self.player_top_left = (-1, -1)
        self.player_bottom_right = (-1, -1)
        
    def _getReward(self):
        playerStars = self.ScreenProcessor.getStars(self.player_top_left, self.player_bottom_right)
        teamStars = self.ScreenProcessor.getTeamStars()
        return playerStars + teamStars

    def step(self, action):
        state = self._getObservation()
        done = self._isDone()
        reward = self._getReward()
        self.time_step += 1
        return state, reward, done

    def reset(self):
        self.time_step = 0
        # Reset the game
        PressKey(B)
        time.sleep(0.1)
        ReleaseKey(B)
        time.sleep(1.5)
        PressKey(B)
        time.sleep(0.1)
        ReleaseKey(B)
        return self._getObservation()

    # TODO: check if the game is over through OCR
    def _isDone(self):
        return

    """
    In addition to the features from the screen,
    the player's stars might be a factor in determining the
    play style (i.e. conservative or aggresive)
    """
    def _getObservation(self):
        # Take a snapshot of the current game screen
        self.ScreenProcessor.orig_screen = ImageGrab.grab(bbox=(0,30,1280,745))
        screen =  np.array(self.ScreenProcessor.orig_screen)
        screen = cv2.resize(screen, (224,224))
        features = self.feature_model.predict([screen.reshape(1,224,224,3)])[0]

        self.player_top_left, self.player_bottom_right = self.ScreenProcessor.getPlayerPosition()
        return features
    