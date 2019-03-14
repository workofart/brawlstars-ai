import numpy as np
import pickle
from PIL import ImageGrab, ImageOps, Image
import cv2
import time
import pyautogui, pytesseract
from utilities.directkeys import PressKey, ReleaseKey, W, A, S, D
from utilities.utilities import mouse
from utilities.img_utils import process_img, match
import imutils
from imutils import contours as ctr_utils

def digit_parser():
    ref = cv2.imread('digits/digits.png')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 200, 255, cv2.THRESH_BINARY_INV)[1]
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = ctr_utils.sort_contours(refCnts, method="left-to-right")[0]
    digits = {}
    # loop over the digit reference contours
    for (i, c) in enumerate(refCnts):
        # compute the bounding box for the digit and extract it
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
    
        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi
    return digits

def process_target_digit(image, refDigits):
    # load the input image, resize it, and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # final = gray.copy() # Used for testing only

    processed_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    new, contours, hierarchy = cv2.findContours(processed_img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = ctr_utils.sort_contours(contours, method="left-to-right")[0]

    # cv2.drawContours(processed_img, digitCnts, -1, (0,255,0), 3)

    groupOutput = []
    digitCnts = []
    
    # loop over the digit contours
    for c in contours:
        # compute the bounding box of the individual digits
        (x, y, w, h) = cv2.boundingRect(np.array(c))

        # Make sure only consider digit size contours, not other noise
        if (w >= 14 and w <= 23) and (h >= 28):
            digitCnts.append([x, y, w, h])
            """
            Below is only for testing, displaying the bounding boxes
            """
            # x1 = x+w
            # y1 = y+h
            # digitCnts.append([x,x1,y,y1])
            # cv2.rectangle(final, (x,y), (x1,y1), (0,255,0),2)
    
    for d in digitCnts:
        (x,y,w,h) = d
        roi = processed_img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (24, 30), interpolation=cv2.INTER_AREA)
        # cv2.imshow('roi', roi)
 
        # initialize a list of template matching scores	
        scores = []
 
        # loop over the reference digits and digit ROI
        for (digit, digitROI) in refDigits.items():
            result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    return groupOutput

def getTeamStars(image, refDigits):
    image = image.crop(box=(90,23,140,60))

    teamstars = ''.join(process_target_digit(np.array(image), refDigits))
    if teamstars != '':
        return int(teamstars)
    else:
        return -1

def getHP(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.adaptiveThreshold(image,255,1,1,11,2)
    image = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.Canny(image, threshold1 = 300, threshold2=450)
    # data = pytesseract.image_to_string(image)
    # print(data)
    cv2.imshow('hp', cv2.resize(image, (400, 540)))
    
def getStars(orig_image, orig_template, top_left, bottom_right):
    img = orig_image.copy().crop(box=(top_left[0]-20,top_left[1]-67,bottom_right[0]+20,bottom_right[1]-15))
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    processed_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
    new, contours, hierarchy = cv2.findContours(processed_img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    groupOutput = []
    StarCnts = []
    # loop over the stars contours
    for c in contours:
        
        # compute the bounding box of the individual stars
        (x, y, w, h) = cv2.boundingRect(np.array(c))

        # Make sure only consider star size contours, not other noise
        if (w >= 5 and w <= 30) and (h <= 30):
            
            StarCnts.append([x, y, w, h])

            """
            Below is only for testing, displaying the bounding boxes
            """
            # x1 = x+w
            # y1 = y+h
            # cv2.rectangle(gray, (x,y), (x1,y1), (0,255,0),2)
    # cv2.imshow('contours', gray)
    return len(StarCnts)

def getPlayerPosition(screen, name_template, previous_top_left, previous_bottom_right, distance_threshold=100):
    # Find current player position
    top_left, bottom_right, _ = match(screen, name_template)

    """
    The below is experimental for stabalizing the player's position
    by constantly tracking the position and preventing flickering to
    another random position
    """
    # top_left_distance = np.sqrt((top_left[0] - previous_top_left[0]) ** 2 + (top_left[1] - previous_top_left[1]) ** 2)
    # bottom_right_distance = np.sqrt((bottom_right[0] - previous_bottom_right[0]) ** 2 + (bottom_right[1] - previous_bottom_right[1]) ** 2)
    # if top_left_distance < distance_threshold:
    #     previous_top_left = top_left
    # else:
    #     top_left = previous_top_left
    # if bottom_right_distance < distance_threshold:
    #     previous_bottom_right = bottom_right
    # else:
    #     bottom_right = previous_bottom_right

    return top_left, bottom_right, previous_top_left, previous_bottom_right

def process_screen():
    refDigits = digit_parser()

    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)

    # last_time = time.time()
    name_template = cv2.imread('object_detection/img/name.png',0)
    stars_template = cv2.imread('object_detection/img/stars.png',1)

    previous_top_left=(485,334)
    previous_bottom_right=(759, 373)
    while True:
        raw_img = ImageGrab.grab(bbox=(0,30,1280,745))
        screen =  np.array(raw_img.copy())
        screen = process_img(screen)

        mouse(screen)

        top_left, bottom_right, previous_top_left, previous_bottom_right = getPlayerPosition(screen, name_template, previous_top_left, previous_bottom_right)
            
        # Get the stars from the stars_img
        playerStars = getStars(raw_img, stars_template, top_left, bottom_right)
        print('Player stars: ' + str(playerStars))

        teamStars = getTeamStars(raw_img, refDigits)
        print('Team stars: ' + str(teamStars))

        # TODO: Crop out the hp img based on player position
        # hp_img = raw_img.crop(box=(top_left[0],top_left[1]+18,bottom_right[0],bottom_right[1]+20)) 
        # getHP(hp_img)
        
        cv2.imshow('window', screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

process_screen()