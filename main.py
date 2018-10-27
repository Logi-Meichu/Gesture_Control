import cv2 as cv
import numpy as np
import json

cap = cv.VideoCapture(0)
bg = cv.createBackgroundSubtractorMOG2(10,20)
detect = False
right_sided = True
count = 0

try:
    fh = open("type.txt","r")
    type = fh.readlines()
    fh.close()
    fh = open("action.txt","r")
    action = fh.readlines()
    fh.close()
except:
    pass

while(True):
    _, frame = cap.read()
    height, width,_ = frame.shape
    lr = -1
    if detect:
        lr = 0
    fgmask = bg.apply(frame, learningRate = lr)
    kernel = np.ones((4,4),np.uint8)
    ret,fgmask = cv.threshold(fgmask,0,255,cv.THRESH_BINARY)
    fgmask = cv.erode(fgmask,kernel,iterations=1)
    fgmask = cv.dilate(fgmask,kernel,iterations=1)
    fgmask = cv.erode(fgmask,kernel,iterations=1)
    fgmask = cv.dilate(fgmask,kernel,iterations=3)
    fgmask = cv.erode(fgmask,kernel,iterations = 1)
    if right_sided:
        cropped_fgmask = fgmask[0:height//5*3,0:width//5*3]
        cropped_frame = frame[0:height//5*3,0:width//5*3]        
    else:
        cropped_fgmask = fgmask[0:height//5*3,width//3:width]
        cropped_frame = frame[0:height//5*3,width//3:width]        
    _,contours, hierarchy = cv.findContours(cropped_fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    black = np.zeros((height//5*3,width//5*2,3))
    cv.drawContours(black, contours, -1, (255,255,255), 3)
    cv.imshow('frame',black)
    k = cv.waitKey(1)
    if k == ord('w'):
        detect = not detect
    if k == ord('f'):
        cv.imwrite("hand_five"+count+".jpg",black) 
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
