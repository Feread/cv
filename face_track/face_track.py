import cv2
import numpy as np
import time
import sys


def store_faces():
    for (x, y, w, h) in faces:
        print(x, y, w, h, len(faces))
        gregion = gframe[y:y+h, x:x+w]
        cregion = cframe[y:y+h, x:x+w]
        global fid
        frame_capture_name = "capture-" + str(fid) + ".png"
        cv2.imwrite(frame_capture_name, cregion)
        fid = fid + 1

        
def show_faces():
    for (x, y, w, h) in faces:
        color = (0, 0, 255)
        stroke = 2
        endx = x + w
        endy = y + h
        cv2.rectangle(cframe, (x, y), (endx, endy), color, stroke)


cascade_frontal_face = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
t_end = time.time() + 5


faces_detected = 0
total_iterations = 0
fid = 0
input()
while (time.time() < t_end):
    ret, cframe = capture.read()
    gframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
  
    faces = cascade_frontal_face.detectMultiScale(gframe, scaleFactor=1.5, minNeighbors=1)
    store_faces()
    show_faces()

    cv2.imshow("Face Tracker", cframe)
    if (len(faces) != 0):
            faces_detected = faces_detected + 1
    total_iterations = total_iterations + 1
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

success_rate = (faces_detected / total_iterations) * 100
sys.stdout.flush()
print("Efficiency rate:", success_rate)
capture.release()
cv2.destroyAllWindows()