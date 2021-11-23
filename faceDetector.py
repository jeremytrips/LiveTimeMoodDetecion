from copy import deepcopy
import cv2
from deepface.commons.functions import preprocess_face
from keras.backend import eye
import numpy as np
from PIL import Image

TARGET_SIZE=(48, 48)

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def process_face(img):
    """
    Function taken from the deepface package
    It is used to pre-process images to the correct size
    """
    #img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    factor_0 = TARGET_SIZE[0] / img.shape[0]
    factor_1 = TARGET_SIZE[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = TARGET_SIZE[0] - img.shape[0]
    diff_1 = TARGET_SIZE[1] - img.shape[1]

    img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE)

    #normalizing the image pixels
    img_pixels = Image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]

    return img_pixels


def detect_faces(img):
    """
    Detect and isolate the head in the img 
    """
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    resp = []
    detected_face = None
    faces = []
    try:
        faces = face_detector.detectMultiScale(img, 1.1, 10)
    except:
        # todo handle the exception
        pass

    if len(faces) > 0:
        for x,y,w,h in faces:
            temp = img[int(y):int(y+h), int(x):int(x+w)]
            detected_face = preprocess_face(temp)
            resp.append((detected_face, [x, y, w, h]))
        return resp

    return None
