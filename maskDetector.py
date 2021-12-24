import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout


## https://colab.research.google.com/drive/1zqwKEe5ibrArYw66DTsgJ5mHBUiZvOGj#scrollTo=gb1D07MBhiWg


def analyze(model, face_img):
    
    # On met le visage a taille pour le reseau
    rerect_sized=cv2.resize(face_img,(224,224))
    normalized=rerect_sized/255.0
    reshaped=np.reshape(normalized,(1,224,224,3))
    reshaped = np.vstack([reshaped])

    # On met le nouveau visage dans le reseau
    result=model.predict(reshaped)

    # on prends la plus grande probabilité des deux
    label=np.argmax(result,axis=1)[0]

    return label