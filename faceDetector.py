import cv2
from keras.preprocessing import image

import numpy as np
import math

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def process_face(img, target_size=(48,48), to_gray=True):	
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    if to_gray:
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
    else:
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

    if to_gray:
        img = np.expand_dims(img, axis = 0)
        img = np.divide(img, 255) 

    return img

def align_face(img):
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #eye detector expects gray scale image
    eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

    if len(eyes) >= 2:

        eye_1 = eyes[0]; eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1; right_eye = eye_2
        else:
            left_eye = eye_2; right_eye = eye_1

        #-----------------------
        #find center of eyes
        left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        
        #this function aligns given face in img based on left and right eye coordinates

        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        #-----------------------
        #find rotation direction

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock

        #-----------------------
        #find length of triangle edges

        a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

        #-----------------------

        #apply cosine rule

        if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

            cos_a = (b*b + c*c - a*a)/(2*b*c)
            angle = np.arccos(cos_a) #angle in radian
            angle = (angle * 180) / math.pi #radian to degree
            if direction == -1:
                angle = 90 - angle
            return direction * angle
        return None
    return None 



def detect_faces(img):
    """
    Detect and isolate the head in the img 
    """
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

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

            rotation = align_face(temp)
            detected_face = process_face(temp)
            resp.append((detected_face, [x, y, w, h], rotation))
        return resp

    return None
