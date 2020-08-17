import cv2
from recognition import *


test_img=cv2.imread('face.jpg')#test_img path
faceDetected,gray = faceDetection(test_img)
print("faces_detected:",faceDetected)

faces,faceID = labels_for_training_data('training')
face_recognizer = train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')
name = {0: 'niyati', 1: 'vandit'}
# for (x,y,w,h) in facesDetected:
#     cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,255,0), 5)
#
# cv2.imshow('img', test_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

test_img = cv2.resize(test_img, (1280, 720))
facesDetected, gray = faceDetection(test_img)
for face in facesDetected:
    (x,y,w,h) = face
    roi_gray = gray[y:y+h, x:x+w]
    labels, confidence = face_recognizer.predict(roi_gray)

    if confidence > 140:
        continue
    draw_rect(test_img, face)
    predicted_name = name[labels]
    put_text(test_img, predicted_name, x, y+100)
    put_text(test_img, str(confidence), x, y + 150)

cv2.imshow('img', test_img)
# if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
#     break
cv2.waitKey()
cv2.destroyAllWindows