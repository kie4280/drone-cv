import cv2
import numpy as np


class HUD:
    def __init__(self) -> None:
        self.data = {}

    def addFields(self, *kargs):
        for i in kargs:
            self.data[i] = "NA"

    def update(self, field, val) -> bool:
        if field not in self.data.keys():
            return False
        self.data[field] = str(val)

    def getFrame(self, frame):

        for i, k in enumerate(self.data.keys()):
            v = self.data[k]
            frame = cv2.putText(frame, k + ": " + v,
                                (0, i * 30+30),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1, (0, 0, 255), 1, cv2.LINE_AA)
        return frame


if __name__ != "__main__":
    h = HUD()
