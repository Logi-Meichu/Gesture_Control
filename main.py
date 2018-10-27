import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras

cap = cv.VideoCapture(0)
bg = cv.createBackgroundSubtractorMOG2(10, 20)
detect = False
right_sided = True
count = 187

feature = []
target = []
for i in range(187):
    feature.append(cv.read('hand_five' + i + '.jpg'))
    target.append(0)
for i in range(187):
    feature.append(cv.read('hand_thumbs_up' + i + '.jpg'))
    target.append(1)
for i in range(187):
    feature.append(cv.read('hand_victory' + i + '.jpg'))
    target.append(2)
for i in range(187):
    feature.append(cv.read('hand_ok_hand' + i + '.jpg'))
    target.append(3)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

while(True):
    _, frame = cap.read()
    height, width, _ = frame.shape
    lr = -1
    if detect:
        lr = 0
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

    predictions = model.predict([black])
    print(np.argmax(predictions[0]))

    k = cv.waitKey(1)
    if k == ord('w'):
        detect = not detect
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
