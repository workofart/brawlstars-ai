from object_detection.processor import ScreenProcessor
import os
import cv2
# p = ScreenProcessor(None)
# p.record_screen()



def convert_BGR2RGB():
    files = []
    path = 'object_detection/raw_screen'
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f, img)

    