import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D
from template_matching import match

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 300, threshold2=500)
    return processed_img

def main():
    
    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)

    # last_time = time.time()
    template = cv2.imread('object_detection/img/name.png',0)
    while True:
        # PressKey(W)
        # ReleaseKey(W)
        raw_img = ImageGrab.grab(bbox=(0,30,1280,745))
        screen =  np.array(raw_img)
        # last_time = time.time()
        new_screen = process_img(screen)
        
        cv2.imshow('window', match(new_screen, template))
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()