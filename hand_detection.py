import numpy as np
import math
import cv2 as cv
from hand_recognition import HandRecognition


handRecognition = HandRecognition()

cap = cv.VideoCapture(0)
pred = "None"
xmin, ymin, xmax, ymax = 100, 100, 300, 300
while True:
    _, img = cap.read()
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
    
    k = cv.waitKey(1)
    if k == 27:
        break
    
    elif k%256 == 32:
        
        crop_img = img[xmin:xmax, ymin:ymax][:, :, 2]
        
        crop_img = cv.resize(crop_img, (100, 100))
        
        pred = handRecognition.predict(crop_img, mode = 'MLP')
        
        
    for i in range(100,300):
        for j in range(100,300):
            # print(img[i][j].tolist().sum())
            if img[i][j].tolist()[0] > 20 and img[i][j].tolist()[0] < 100:
                img = cv.circle(img, (j,i), radius=0, color=(255, 0, 0), thickness=-1)
        
    cv.putText(img, str(pred), (550, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('main window', img)