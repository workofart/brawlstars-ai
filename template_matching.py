import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def example():
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

def match(img, template):
    template = cv.Canny(template, threshold1 = 300, threshold2=500)
    # cv.imshow("Template", template)
    h, w = [template.shape[0], template.shape[1]]
    method = cv.TM_CCORR
    # img = orig_image.copy()
    
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    
    # Get the box coordinate from the matched result
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Match Multiple Occurences
    # loc = np.where(res >= 0.5)
    # for pt in zip(*loc[::-1]):
    #     cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    # return img

    cv.rectangle(img,top_left, bottom_right, 255, 2)
    return top_left, bottom_right

