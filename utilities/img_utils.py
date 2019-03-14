import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_img(image, t1=150,t2=300):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img =  cv2.Canny(processed_img, threshold1=t1, threshold2=t2)
    return processed_img

def template_match_example():
    img = cv.imread('object_detection/img/train/020.png',0)
    img2 = img.copy()
    template = cv.imread('object_detection/img/name.png',0)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()

def match(img, template, isCanny = True, method=cv.TM_CCORR):
    if isCanny:
        template = cv.Canny(template, threshold1 = 150, threshold2=300)
    
    # cv.imshow("Template", template)
    h, w = [template.shape[0], template.shape[1]]
    
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    
    # Get the box coordinate from the matched result
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(img,top_left, bottom_right, 255, 2)
    return top_left, bottom_right, res

