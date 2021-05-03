import cv2 
import numpy as np


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def cal():
    pass

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("should not happen")
    while True:
        ret, frame = cap.read()
        cv2.imshow("sdf", frame)
        cv2.waitKey(1)
    pass