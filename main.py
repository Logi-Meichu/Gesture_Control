import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import win32com.client as comclt
from win32gui import GetWindowText, GetForegroundWindow
import pyautogui


shell = comclt.Dispatch("WScript.Shell")

cap = cv.VideoCapture(0)
bg = cv.createBackgroundSubtractorMOG2(100, 20)
detect = False
right_sided = True
count = 264
cnt = 0
pre = 1
delay = -1

feature = []
target = []

for i in range(count):
    feature.append(cv.imread('hand_five' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE))
    target.append(0)
# for i in range(count):
#     feature.append(cv.imread('hand_thumbs_up' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE))
#     target.append(1)
for i in range(count):
    feature.append(cv.imread('hand_nothing' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE))
    target.append(1)
h,w = feature[0].shape

feature = np.asarray(feature)
print(feature.shape)
target = np.asarray(target)
feature = feature / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(h,w)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(feature, target, epochs=2)

while(True):
    _, frame = cap.read()
    height, width, _ = frame.shape
    lr = -0.001
    fgmask = bg.apply(frame, learningRate=lr)
    kernel = np.ones((4, 4), np.uint8)
    ret, fgmask = cv.threshold(fgmask, 0, 255, cv.THRESH_BINARY)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    fgmask = cv.dilate(fgmask, kernel, iterations=1)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    fgmask = cv.dilate(fgmask, kernel, iterations=3)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    if right_sided:
        cropped_fgmask = fgmask[0:height // 4 * 3, 0:width // 5 * 2]
        cropped_frame = frame[0:height // 4 * 3, 0:width // 5 * 2]
    else:
        cropped_fgmask = fgmask[0:height // 4 * 3, width // 5 * 2:width]
        cropped_frame = frame[0:height // 4 * 3, width // 5 * 2:width]
    _, contours, hierarchy = cv.findContours(
        cropped_fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    black = np.zeros((height // 4 * 3, width // 5 * 2, 3))
    cv.drawContours(black, contours, -1, (255, 255, 255), 3)
    cv.imshow('frame', black)
    
    black = np.uint8(black)
    black=cv.cvtColor(black,code=cv.COLOR_RGB2GRAY)
    predictions = model.predict(np.asarray([black]))
    cur = np.argmax(predictions[0])
    if delay == -1:
        if pre == 0:
            if cur == 0:
                count = count + 1
                if count > 20:
                    if GetWindowText(GetForegroundWindow()) == "Killing Floor 2 (64-bit, DX11) v1070":
                        shell.SendKeys("u")
                    if GetWindowText(GetForegroundWindow()) == "BattleBlock Theater":
                        pyautogui.keyDown('ctrlleft')
                    count = 0
                    delay = 0
            else:
                if count > 20:
                    if GetWindowText(GetForegroundWindow()) == "Killing Floor 2 (64-bit, DX11) v1070":
                        shell.SendKeys("u")
                    if GetWindowText(GetForegroundWindow()) == "BattleBlock Theater":
                        pyautogui.keyDown('ctrlleft')
                    count = 0
                    delay = 0
                else:
                    if GetWindowText(GetForegroundWindow()) == "Killing Floor 2 (64-bit, DX11) v1070":
                        shell.SendKeys("i")
                    if GetWindowText(GetForegroundWindow()) == "BattleBlock Theater":
                        pyautogui.keyDown('e')
                    count = 0
                    delay = 0
        else:
            if cur == 0:
                count = count + 1
    else:
        delay = delay + 1
        if delay == 45:
            pyautogui.keyUp('ctrlleft')
            pyautogui.keyUp('e')
            delay = -1
    pre = cur
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
