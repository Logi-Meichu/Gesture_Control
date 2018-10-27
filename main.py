import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
bg = cv.createBackgroundSubtractorMOG2(10,20)
detect = False
right_sided = True

while(True):
    _, frame = cap.read()
    height, width,_ = frame.shape
    lr = -1
    if detect:
        lr = 0
    fgmask = bg.apply(frame, learningRate = lr)
    kernel = np.ones((4,4),np.uint8)
    ret,fgmask = cv.threshold(fgmask,0,255,cv.THRESH_BINARY)
    if right_sided:
        cropped_fgmask = fgmask[0:height//5*3,0:width//2]
        cropped_frame = frame[0:height//5*3,0:width//2]        
    else:
        cropped_fgmask = fgmask[0:height//5*3,width//2:width]
        cropped_frame = frame[0:height//5*3,width//2:width]        
    _,contours, hierarchy = cv.findContours(cropped_fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(cropped_frame, contours, -1, (0,255,0), 3)
    #cv.imshow('frame',cropped_frame)
    cv.imshow('frame',fgmask)


    k = cv.waitKey(1)
    if k == ord('w'):
        detect = not detect
    if k == 27:
        break



cap.release()
cv.destroyAllWindows()
