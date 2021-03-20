from typing import Counter
import cv2
import numpy as np

# connect -2 means conflict


def getConnected(fore, threshold=20):

    connect = np.zeros((fore.shape[0], fore.shape[1]), dtype=int)
    cat_counter: int = 0
    area_counter: dict = dict()
    index_mapping: dict = dict()

    it = np.nditer(fore, flags=['multi_index'], op_flags=['readonly'])
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
                index_mapping[max(left, top)] = index_mapping[min(left, top)]
                area_counter[index_mapping[top]] += 1
            else:
                connect[cur_pos] = left
                area_counter[index_mapping[left]] += 1
        else:
            if left != 0:
                connect[cur_pos] = left
                area_counter[index_mapping[left]] += 1
            else:
                connect[cur_pos] = top
                area_counter[index_mapping[top]] += 1
        # print(it.multi_index)

    it = np.nditer(connect, flags=['multi_index'], op_flags=['readwrite'])
    for x in it:
        cur_pos = it.multi_index
        left = 0 if cur_pos[1] == 0 else connect[cur_pos[0], cur_pos[1]-1]
        top = 0 if cur_pos[0] == 0 else connect[cur_pos[0]-1, cur_pos[1]]
        
        if x != 0 :
            if area_counter[index_mapping[int(x)]] > threshold:
                x[...] = index_mapping[int(x)]
            else:
                x[...] = 0
    return connect


def getBound(connect):
    bounds = dict()
    it = np.nditer(connect, flags=['multi_index'], op_flags=['readwrite'])
    for x in it:
        if x == 0:
            continue
        x = int(x)
        if x not in bounds.keys():
            bounds[x] = [10000, 10000, 0, 0]
        else:
            bounds[x] = [min(bounds[x][0], it.multi_index[1]), min(bounds[x][1], it.multi_index[0]), max(
                bounds[x][2], it.multi_index[1]), max(bounds[x][3], it.multi_index[0])]
    return bounds


def getFore(frame, backSub):

    fgmask = backSub.apply(frame)
    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)
    # assert ret == True
    return nmask


if __name__ == "__main__":
    cap = cv2.VideoCapture('lab3/vtest.avi')
    backSub = cv2.createBackgroundSubtractorMOG2()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break


        fore = getFore(frame, backSub)
        connected = getConnected(fore, 100)
        ff = getBound(connected)
        for i in ff:
            array = ff[i]
            frame = cv2.rectangle(
                frame, (array[0], array[1]), (array[2], array[3]), color=(255, 0, 0), thickness=2)
        buf = np.zeros(frame.shape, dtype=np.uint8)
        buf[:, :, 2] = connected*30


        cv2.imshow("frame", frame)
        cv2.imshow("connected", buf)
        cv2.imshow("fore", fore)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
