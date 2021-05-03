import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

winStride = (8,8)
scale = 1.2


def getRect(frame):
    rects, weights = hog.detectMultiScale(
        frame, winStride=winStride, scale=scale, useMeanshiftGrouping=False)
    print(weights)

def getFaces(frame):
    face_rects = detector(frame, 0)
    for i, d in enumerate(face_rects):
        frame = cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)
    return frame
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("should not happen")
    while True:
        ret, frame = cap.read()
        # getRect(frame)
        frame = getFaces(frame)
        cv2.imshow("sdf", frame)
        cv2.waitKey(1)
    pass
