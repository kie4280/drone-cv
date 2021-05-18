import cv2
import numpy as np

video_path = "rtsp://192.168.0.102:5554/h264"
cap = cv2.VideoCapture(video_path)

assert cap.isOpened()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("sdf", frame)
    cv2.waitKey(1)