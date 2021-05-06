import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

winStride = (8, 8)
scale = 1.2


def getHOG(frame):
    rects, weights = hog.detectMultiScale(
        frame, winStride=winStride, scale=scale, useMeanshiftGrouping=False)
    for i, d in enumerate(rects):

        frame = cv2.rectangle(
            frame, (d[0], d[1]), (d[0]+d[2], d[1]+d[3]), (0, 255, 0), 2)
    # frame = cv2.rectangle(frame, [rects[0:2]], [rects[2:]], (0,0,255), 2)
        distance = 140000 / d[3]
        frame = cv2.putText(frame, 'people ' + str("{:.2f}".format(distance)),
                            ((d[0] - 30),
                             (d[1] + 50)), cv2.FONT_HERSHEY_DUPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        print("people distance", distance)

    return frame


def getFaces(frame):
    face_rects = detector(frame, 0)
    for i, d in enumerate(face_rects):
        face_width = d.right() - d.left()
        face_height = d.bottom() - d.top()
        distance = 9000 / ((face_width + face_height) / 2)
        print("face distance", distance)
        frame = cv2.rectangle(frame, (d.left(), d.top()),
                              (d.right(), d.bottom()), (100, 255, 0), 2)
        frame = cv2.putText(frame, 'face ' + str("{:.2f}".format(distance)),
                            ((d.right() + d.left()) // 2 - 50,
                             (d.top() + d.bottom()) // 2), cv2.FONT_HERSHEY_DUPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("should not happen")
    while True:
        ret, frame = cap.read()
        frame = getHOG(frame)
        frame = getFaces(frame)
        cv2.imshow("sdf", frame)
        cv2.waitKey(1)
    pass
