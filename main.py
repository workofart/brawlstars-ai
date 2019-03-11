import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui, pytesseract
from directkeys import PressKey, ReleaseKey, W, A, S, D
from template_matching import match

STABLE_FACTOR = 0.1

def getHP(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.adaptiveThreshold(image,255,1,1,11,2)
    image = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.Canny(image, threshold1 = 300, threshold2=450)
    # data = pytesseract.image_to_string(image)
    # print(data)
    cv2.imshow('hp', cv2.resize(image, (400, 540)))
    
def getStars(orig_image, template, top_left, bottom_right):
    img = orig_image.copy()
    # Crop out the stars img based on player positon
    stars_img = img.crop(box=(top_left[0]-20,top_left[1]-67,bottom_right[0]+20,bottom_right[1]-15))
    stars_img = process_img(np.array(stars_img))

    template = cv2.Canny(template, threshold1 = 300, threshold2=500)
    h, w = [template.shape[0], template.shape[1]]
    
     # Match Multiple Occurences
    res = cv2.matchTemplate(stars_img,template,cv2.TM_SQDIFF)
    loc = np.where(res >= 0.999)
    counter = 0
    for pt in zip(*loc[::-1]):
        cv2.rectangle(stars_img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        counter += 1
    # print('counter' + str(counter))
    # print(len(loc[::-1]))
    # cv2.imshow('stars', cv2.resize(stars_img, (400, 540)))
    # cv2.imshow('stars', stars_img)


def process_img(image):
    original_image = image
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_img =  cv2.Canny(processed_img, threshold1 = 300, threshold2=500)
    return processed_img

def main():
    
    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)

    # last_time = time.time()
    name_template = cv2.imread('object_detection/img/name.png',0)
    stars_template = cv2.imread('object_detection/img/stars.png',0)

    previous_top_left = (-1,-1)
    previous_bottom_right = (-1, -1)
    previous_top_left_distance = 420
    previous_bottom_right_distance = 420

    while True:
        raw_img = ImageGrab.grab(bbox=(0,30,1280,745))
        screen =  np.array(raw_img)
        screen = process_img(screen)

        # Find current player position
        top_left, bottom_right = match(screen, name_template)
        top_left_distance = np.sqrt((top_left[0] - previous_top_left[0]) ** 2 + (top_left[1] - previous_top_left[1]) ** 2)
        bottom_right_distance = np.sqrt((bottom_right[0] - previous_bottom_right[0]) ** 2 + (bottom_right[1] - previous_bottom_right[1]) ** 2)
        
        # TODO: After the initial distance calculation, the previous distance is set to be very large, which then makes it impossible to set back
        if (top_left_distance > previous_top_left_distance * (1+STABLE_FACTOR) or top_left_distance < previous_top_left_distance * (1-STABLE_FACTOR)) and previous_top_left_distance != 0:
            top_left = previous_top_left
        else:
            print(top_left_distance)
            previous_top_left = top_left
            previous_top_left_distance = top_left_distance

        if (bottom_right_distance > previous_bottom_right_distance * (1-STABLE_FACTOR) or bottom_right_distance < previous_bottom_right_distance * (1-STABLE_FACTOR)) and previous_bottom_right_distance != 0:
            bottom_right = previous_bottom_right
        else:
            print(bottom_right_distance)
            previous_bottom_right = bottom_right
            previous_bottom_right_distance = bottom_right_distance

        print('---')
        # Get the stars from the stars_img
        # getStars(raw_img, stars_template, top_left, bottom_right)

        # Crop out the hp img based on player position
        # hp_img = raw_img.crop(box=(top_left[0],top_left[1]+18,bottom_right[0],bottom_right[1]+20)) 
        # getHP(hp_img)
        
        cv2.imshow('window', screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()