import cv2
import numpy as np

def getConnected(frame):
    mask = getFore (frame)
    connect= np.zeros(frame.shape,dtype=np.int)
    it = np.nditer(mask, flags=['multi_index'], op_flags=['readonly'])
	pixel_num= frame.shape[0]*frame.shape[1]
	label = [[0] * (pixel_num/10)] * (pixel_num/10)
	num_label = 0
	num_pixel = [0] * (pixel_num/10)
    for x in it:
        cur = it.multi_index()
        if cur[0] == 0:
            if cur[1] == 0:
                label[0][0]=cur
				num_pixel[0]=1
				num_label=1
            else:
                if frame[cur[0]][cur[1]-1]!=255:
					frame[cur[0]][cur[1]]= frame[cur[0]][cur[1]-1]
					label[frame[cur[0]][cur[1]]][num_pixel[frame[cur[0]][cur[1]]]=cur
					num_pixel[frame[0][cur[1]]]+=1
				else:
					label[num_label][0]=cur
					frame[cur[0]][cur[1]]= num_label
					num_pixel[num_label]+=1
					num_label+=1
				
		elif cur[1]==0:
			if frame[cur[0]-1][cur[1]]!=255:
				frame[cur[0]][cur[1]]= frame[cur[0]-1][cur[1]]
				label[frame[cur[0]][cur[1]]][num_pixel[frame[cur[0]][cur[1]]]=cur
				num_pixel[frame[0][cur[1]]]+=1
			else:
				label[num_label][0]=cur
				frame[cur[0]][cur[1]]= num_label
				num_pixel[num_label]+=1
				num_label+=1
		else:
			if frame[cur[0]-1][cur[1]]!=255:
				frame[cur[0]][cur[1]]= frame[cur[0]-1][cur[1]]
				label[frame[cur[0]][cur[1]]][num_pixel[frame[cur[0]][cur[1]]]=cur
				num_pixel[frame[0][cur[1]]]+=1

				if frame[cur[0]][cur[1]-1]!=255:
					label[cur[0]-1][cur[1]].extend(label[frame[cur[0]][cur[1]-1]])
					del label[frame[cur[0]][cur[1]-1]

			elif frame[cur[0]][cur[1]-1]!=255:
				frame[cur[0]][cur[1]]= frame[cur[0]][cur[1]-1]
				label[frame[cur[0]][cur[1]]][num_pixel[frame[cur[0]][cur[1]]]=cur
				num_pixel[frame[0][cur[1]]]+=1
			else:
				label[num_label][0]=cur
				frame[cur[0]][cur[1]]= num_label
				num_pixel[num_label]+=1
				num_label+=1
print(label)
        

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
        waitKey()
    cv2.destroyAllWindows()
