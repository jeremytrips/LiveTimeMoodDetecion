import numpy as np
import cv2

import faceDetector
import moodDetector

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

while 1:
    ret, img = cap.read()
    faces = faceDetector.detect_faces(img)
    if faces is not None:
        for item in faces:
            res = moodDetector.analyze(item[0])
            print(res)
            (x,y,w,h) = item[1]
            cv2.rectangle(img, (x, y),(x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img',img) 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()