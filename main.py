import numpy as np
import cv2
from matplotlib import pyplot as plt

vid = cv2.VideoCapture("Road.mov")

while (True):
    ret, img = vid.read()

    # CONTRAST ENHANCEMENT
    def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
        brightness += int(round(255 * (1 - contrast) / 2))
        return cv2.addWeighted(img, contrast, img, 0, brightness)

    contrast_img = adjust_contrast_brightness(img, 15.0, -900) # contrast then brightness

    # YELLOW LINE ENHANCEMENT
    lower_yellow = np.array([0, 0, 150])
    upper_yellow = np.array([10, 10, 255])
    yellow_mask = cv2.inRange(contrast_img, lower_yellow, upper_yellow)

    # WHITE LINE ENHANCEMENT
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(contrast_img, lower_white, upper_white)

    # ADDING MASKS
    dst = cv2.add(white_mask, yellow_mask)

    # PERSPECTIVE TRANSFORM
    pts1 = np.float32([[480, 1250], [620, 1250], [150, 1550], [950, 1550]])
    pts2 = np.float32([[0, 0], [475, 0], [0, 640], [475, 640]])
    pts3 = np.array(pts1, np.int32)
    #cv2.fillPoly(img, pts=[pts3], color=(255,0,0))

    # PERSPECTIVE WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(dst, matrix, (500, 600))

    # SHOW IMAGE
    cv2.imshow('frame', result)
    cv2.imshow('frame1', img)

    # HISTOGRAM
    #plt.hist(result.ravel(),256,[0,256]); plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
