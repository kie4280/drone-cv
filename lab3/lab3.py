from typing import Counter
import cv2
import numpy as np
from numpy.core.defchararray import index

# connect -2 means conflict

def getConnected(frame):
    mask = getFore (frame)
    connect= np.zeros((frame.shape[0], frame.shape[1]),dtype=np.int)
    cat_counter:int = 0
    area_counter:dict = dict()
    index_mapping:dict = dict()
    
    it = np.nditer(mask, flags=['multi_index'], op_flags=['readonly'])
    for x in it:
        cur_pos = it.multi_index
        left = 0 if cur_pos[1] == 0 else connect[cur_pos[0], cur_pos[1]-1]
        top = 0 if cur_pos[0] == 0 else connect[cur_pos[0]-1, cur_pos[1]]

        if x == 0:
            continue
        if left == 0 and top == 0:
            cat_counter += 1
            connect[cur_pos] = cat_counter
            index_mapping[cat_counter] = cat_counter
        elif left != 0 and top != 0:
            if left != top:
                connect[cur_pos] = -2
            else:
                connect[cur_pos] = left
        else:
            if left != 0:
                connect[cur_pos] = left
            else:
                connect[cur_pos] = top
        print(it.multi_index)

    it = np.nditer(connect, flags=['multi_index'], op_flags=['readonly'])
    for x in it:
        cur_pos = it.multi_index
        left = 0 if cur_pos[1] == 0 else connect[cur_pos[0], cur_pos[1]-1]
        top = 0 if cur_pos[0] == 0 else connect[cur_pos[0]-1, cur_pos[1]]
        if x == -2:
            m1 = 0;m2 = 0
            if left < top:
                m1 = left
                m2 = top
            else :
                m1 = top
                m2 = left
            index_mapping[m2] = m1

def getFore(frame):
    backSub = cv2.createBackgroundSubtractorMOG2()
    fgmask = backSub.apply(frame)
    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)
    # assert ret == True
    return nmask

if __name__ == "__main__":
    cap = cv2.VideoCapture('lab3/vtest.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        # cv2.imshow("frame",getFore(frame))
        # cv2.waitKey(33)
        getConnected(frame)
    cv2.destroyAllWindows()
