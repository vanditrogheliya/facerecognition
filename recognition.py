import os
import cv2
import numpy as np

def faceDetection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_harcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_harcascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)
    return faces, gray_img

def labels_for_training_data(dictionary):
    faces = []
    faceID = []

    for path, subdir, filenames in os.walk(dictionary):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print('img+path: ', img_path)
            print('id: ', id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print('image not loaded')
                continue
            faces_rect, gray_img = faceDetection(test_img)
            if len(faces_rect) != 1:
                continue
            (x,y,w,h) = faces_rect[0]
            roi_gray = gray_img[y:y+h, x:x+w]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def draw_rect(test_img, face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,255,0), 3)

def put_text(test_img, text, x,y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255,255,255), 2)






