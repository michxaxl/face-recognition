import cv2
import csv
import numpy as np
import FaceDetect as fd
faces_loaded, labels_loaded = fd.BaseLoad("test_s3.csv")
faces = []
labels = []

    # print(type(faces_loaded[0]))
    # len(faces_loaded)
for x in range(len(faces_loaded)):
    if type(fd.FaceDetect(faces_loaded[x])) != type(None):
        print(type(fd.FaceDetect(faces_loaded[x])))
        faces.append(fd.FaceDetect(faces_loaded[x]))
        labels.append(int(labels_loaded[x], base=8))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.write("reco_3.xml")