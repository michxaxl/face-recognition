import cv2
import csv
import numpy as np
import FaceDetect as fd
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

test_img = cv2.imread("lena.png", 1)
test2_img = cv2.imread("ExtendedYaleB/s2/yaleB11/yaleB11_P00A+000E+45.pgm",1)
test3_img = cv2.imread("ExtendedYaleB/s2/yaleB12/yaleB12_P00A+000E+45.pgm",1)


#faces = face_cascade.detectMultiScale(img, 1.1, 4)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("reco_3.xml")


# make a copy of the image as we don't want to change original image
img = test_img.copy()
img2 = test2_img.copy()
# detect face from the image
face = fd.FaceDetect(img)
face2 = fd.FaceDetect(img2)
face3 = fd.FaceDetect(test3_img)
# predict the image using our face recognizer
label = face_recognizer.predict(face)
print("to jest warostsc predykcji dla YaleB11")
print(label)

label2 = face_recognizer.predict(face2)
print("to jest warostsc predykcji dla leny")
print(label2)

label3 = face_recognizer.predict(face3)
print("to jest warostsc predykcji dla yYleb12")
print(label3)

if label == 0:
    label_text = "monobrew"
else:
    label_text = "nieznane"

if label2 == 0:
    label2_text = "monobrew"
else:
    label2_text = "nieznane"

# draw a rectangle around face detected

# draw name of predicted person
print(label_text)
print(label2_text)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


