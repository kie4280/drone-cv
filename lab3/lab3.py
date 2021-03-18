import cv2
import numpy as np

def getConnected(frame):
    mask = getFore (frame)
    connect= np.zeros(frame.shape,dtype=np.int)
    cat_counter:int = 1
    area_counter:dict = dict()
    
    it = np.nditer(mask, flags=['multi_index'], op_flags=['readonly'])
    for x in it:
        
        cur = it.multi_index()
        if cur[0] == 0:
            if cur[1] == 0:
                connect[cur] = 1
            else:
                if connect[cur[1]-1] == cat_counter:
                    connect[cur] = cat_counter
        print(it.multi_index)

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
