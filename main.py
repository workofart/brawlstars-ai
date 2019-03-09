import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img

def main():
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    for i in range(500):
        # PressKey(W)
        # ReleaseKey(W)
        screen =  np.array(ImageGrab.grab(bbox=(0,30,1280,745)))
        # last_time = time.time()
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        # pyautogui.screenshot("img/showdown/gameplay_{}.png".format(i), region=(0,0,1280,800))
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        time.sleep(3)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()