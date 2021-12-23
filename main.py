import cv2
import numpy as np
import faceDetector
import maskDetector
import moodDetector
from keras.models import load_model
import time

from PIL import Image

emojis = {
    'angry': cv2.imread("./emoji/angry.png"),
    'disgust': cv2.imread("./emoji/disgust.png"),
    'fear': cv2.imread("./emoji/fear.jpeg"),
    'happy': cv2.imread("./emoji/happy.jpeg"),
    'sad': cv2.imread("./emoji/sad.jpeg"),
    'surprise': cv2.imread("./emoji/surprise.jpeg"),
    'neutral': cv2.imread("./emoji/neutral.jpeg"),
    'mask': cv2.imread("./emoji/mask.jpeg")
}


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("./juetmoi.jpg")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

MODEL = moodDetector.loadModel()
MODEL_MASK = load_model("./data/mask_detector.model")

while 1:
    ret, img = cap.read()
    
    faces = faceDetector.detect_faces(img, face_cascade)
    if faces is not None:
        for item in faces:
            mask = maskDetector.analyze(MODEL_MASK, item[1])
            if not mask:
                # https://data-flair.training/blogs/face-mask-detection-with-python/
                mood = "mask"
            else:
                mood = moodDetector.analyze(MODEL, item[0])
            (x,y,w,h) = item[2]

            emoji = emojis[mood]
            # Check if the rotation has been calculated
            if item[3] is not None:
                
                emoji = Image.fromarray(emoji)
                
                emoji = np.array(emoji.rotate(int(-item[3])))

            # formatte l'emoji exactement à la taille de la tête détectée
            emoji = faceDetector.process_face(emoji, target_size=(w, h), to_gray=False)
            img[y:y+h, x:x+w, :] = emoji

    cv2.imshow('img',img) 

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()