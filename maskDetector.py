import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout


## https://colab.research.google.com/drive/1zqwKEe5ibrArYw66DTsgJ5mHBUiZvOGj#scrollTo=gb1D07MBhiWg


def load_model():
    model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),

    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(100, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(100, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.load_weights('./data/model6-004.h5')
    return model

MODEL = load_model()

def analyze(face_img):
    
    # On met le visage a taille pour le reseau
    rerect_sized=cv2.resize(face_img,(150,150))
    normalized=rerect_sized/255.0
    reshaped=np.reshape(normalized,(1,150,150,3))
    reshaped = np.vstack([reshaped])

    # On met le nouveau visage dans le reseau
    result=MODEL.predict(reshaped)

    # on prends la plus grande probabilité des deux
    label=np.argmax(result,axis=1)[0]

    return label