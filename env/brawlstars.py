import numpy as np
from net.mobilenet import mnet_feature
from PIL import ImageGrab
from object_detection.processor import ScreenProcessor, match
from utilities.directkeys import PressKey, ReleaseKey, B
from utilities.utilities import mouse
from experiencebuffer import Experience_Buffer
import cv2
import time
from scipy.stats import mode
from utilities.window import WindowMgr

# w = WindowMgr()
class Brawlstars():

    def __init__(self):
        self.feature_model = mnet_feature()
        self.observation_space = np.zeros((1, 62720)) # TODO: decide on data structure, refer to the feature output from mobilenet
        self.action_space = [0,1,2] # attack, super, no-op
        self.movement_space = [0,1,2,3,4] # left, front, right, back, no-op
        self.ScreenProcessor = ScreenProcessor(ImageGrab.grab(bbox=(0,30,1280,745)))
        self.reward_buffer = Experience_Buffer(10) # This is used to prevent screen flickering skewing rewards
        self.reset()
        
    def _getReward(self):
        playerStars = self.ScreenProcessor.getStars(self.player_top_left, self.player_bottom_right)
        # teamStars = self.ScreenProcessor.getTeamStars()
        
        # Invalid player reward
        if (playerStars == 0 or playerStars > 7):
            pass
        else:
            self.reward_buffer.add(playerStars-2)
        
        mean_reward = np.mean(self.reward_buffer.buffer)
        return mean_reward

    def step(self):
        state = self._getObservation()
        done = self._isDone()
        reward = self._getReward()
        self.time_step += 1
        return state, reward, done

    def reset(self):
        self.time_step = 0
        self.player_top_left = (-1, -1)
        self.player_bottom_right = (-1, -1)
        return self._getObservation()

    def _isDone(self):
        isDone = self.ScreenProcessor.isDone()
        if isDone:
            # print('Restarting game after timestep: {}'.format(self.time_step))
            # Reset the game
            # w.find_window_wildcard("MEmu")
            # w.set_foreground()
            PressKey(B)
            time.sleep(0.3)
            ReleaseKey(B)
            PressKey(B)
            time.sleep(0.3)
            ReleaseKey(B)
            time.sleep(3)
            PressKey(B)
            time.sleep(0.3)
            ReleaseKey(B)
            time.sleep(8) # Wait for game to begin
        return isDone
        # cv2.imshow('done', np.array(self.ScreenProcessor.orig_screen.crop(box=(983,621,1252,689))))
        # cv2.waitKey()

    """
    In addition to the features from the screen,
    the player's stars might be a factor in determining the
    play style (i.e. conservative or aggresive)
    """
    def _getObservation(self):
        # Take a snapshot of the current game screen
        self.ScreenProcessor.orig_screen = ImageGrab.grab(bbox=(0,30,1280,745))
        screen = np.array(self.ScreenProcessor.orig_screen)
        # cv2.imshow('window', screen)
        # mouse(screen)
        # cv2.waitKey()
        screen = cv2.resize(screen, (224,224))
        
        features = self.feature_model.predict([screen.reshape(1,224,224,3)])[0]

        self.player_top_left, self.player_bottom_right = self.ScreenProcessor.getPlayerPosition()
        return features.reshape((1, -1))