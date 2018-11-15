import cv2
import numpy as np


def store_faces():
    id = 0
    for (x, y, w, h) in faces:
        print(x, y, w, h, len(faces))
        gregion = gframe[y:y+h, x:x+w]
        cregion = cframe[y:y+h, x:x+w]
        frame_capture_name = "./faces/capture-" + str(id) + ".png"
        cv2.imwrite(frame_capture_name, cregion)
        id = id + 1

        
def show_faces():
    for (x, y, w, h) in faces:
        color = (0, 0, 255)
        stroke = 2
        endx = x + w
        endy = y + h
        cv2.rectangle(cframe, (x, y), (endx, endy), color, stroke)


cascade_frontal_face = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)

while (True):
    ret, cframe = capture.read()
    gframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
  
    faces = cascade_frontal_face.detectMultiScale(gframe, scaleFactor=1.5, minNeighbors=1)
    store_faces()
    show_faces()

    cv2.imshow("Face Tracker", cframe)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()