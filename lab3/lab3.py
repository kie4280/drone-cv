from typing import Counter
import cv2
import numpy as np
from numpy.core.defchararray import index

# connect -2 means conflict


def getConnected(frame):
    mask = getFore(frame)
    connect = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.int)
    cat_counter: int = 0
    area_counter: dict = dict()
    index_mapping: dict = dict()

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
            area_counter[cat_counter] = 1
        elif left != 0 and top != 0:
            if left != top:
                connect[cur_pos] = min(left, top)
                index_mapping[max(left, top)] = min(left, top)
            else:
                connect[cur_pos] = left
                area_counter[left] = area_counter[left] + 1
        else:
            if left != 0:
                connect[cur_pos] = left
                area_counter[left] = area_counter[left] + 1
            else:
                connect[cur_pos] = top
                area_counter[top] = area_counter[top] + 1
        # print(it.multi_index)

    it = np.nditer(connect, flags=['multi_index'], op_flags=['readwrite'])
    for x in it:
        cur_pos = it.multi_index
        left = 0 if cur_pos[1] == 0 else connect[cur_pos[0], cur_pos[1]-1]
        top = 0 if cur_pos[0] == 0 else connect[cur_pos[0]-1, cur_pos[1]]
        if x != 0:
            x[...] = index_mapping[int(x)]
    return connect


def getArea(connect):
    it = np.nditer(connect, flags=['multi_index'], op_flags=['readwrite'])
    indexs = dict()
    for x in it:
        if x == 0:
            continue
        if indexs[int(x)] == None:
            indexs[int(x)] = 1
        else:
            indexs[int(x)] += 1
    return indexs


def getBound(connect):
    bounds = dict()
    it = np.nditer(connect, flags=['multi_index'], op_flags=['readwrite'])
    for x in it:
        if x == 0:
            continue
        x = int(x)
        if x not in bounds.keys():
            bounds[x] = [1000, 1000, 0, 0]
        else:
            bounds[x] = [min(bounds[x][0], it.multi_index[0]), min(bounds[x][1], it.multi_index[1]), max(
                bounds[x][2], it.multi_index[0]), max(bounds[x][3], it.multi_index[1])]
    return bounds


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
        
        connected = getConnected(frame)

        ff = getBound(connected)
        for i in ff:
            array = ff[i]
            frame = cv2.rectangle(
                frame, array[0], array[1], array[2], array[3], )
        buf = np.zeros(frame.shape, dtype=np.uint8)
        buf[:, :, 2] = connected*30

        cv2.imshow("frame", buf)
        cv2.waitKey(33)
    cv2.destroyAllWindows()
