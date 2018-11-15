import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

while (True):
    ret, video_frame = video_capture.read()
    cv2.imshow("Face Tracker", video_frame)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()