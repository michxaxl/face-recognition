import cv2
import csv

def FaceDetect(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    if (len(faces) == 0):
        return None
    (x,y,w,h) = faces[0]
    return gray_img[y:y+w, x:x+h]

def BaseLoad(base):
    faces=[]
    labels=[]
    with open(base) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            # print(row[0], row[1])
            tmp = cv2.imread(row[0], 1)
            faces.append(tmp)
            labels.append(row[1])

    return faces,labels

def train_recognizer(faces,labels):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


