import cv2
import numpy as np


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

winStride = 10
scale = 10


def getRect(frame):
    rects, weights = hog.detectMultiScale(
        frame, winStride, scale, useMeanshiftGrouping=False)
    print(rects)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("should not happen")
    while True:
        ret, frame = cap.read()
        getRect(frame)
        cv2.imshow("sdf", frame)
        cv2.waitKey(1)
    pass
