import numpy as np
import math
import cv2
from hand_recognition import HandRecognition
import matplotlib.pyplot as plt

handRecognition = HandRecognition()
cap = cv2.VideoCapture(0)
pred = "None"
xmin, ymin, xmax, ymax = 100, 100, 300, 300
while True:
    _, img = cap.read()
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
    
    k = cv2.waitKey(1)
    if k == 27:
        break
    
    elif k%256 == 32:
        
        crop_img = img[xmin:xmax, ymin:ymax][:, :, 0]
        
        crop_img = cv2.resize(crop_img, (100, 100))
        pred = handRecognition.predict(crop_img, mode = 'ConSim')
        
    cv2.putText(img, str(pred), (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('main window', img)