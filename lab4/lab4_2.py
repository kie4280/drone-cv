import cv2 
import numpy as np

p1 = (202, 246)
p2 = (246, 462)
p3 = (631, 260)
p4 = (767, 433)

       
img = cv2.imread('warp.jpg',cv2.IMREAD_COLOR)  

img_height, img_width= np.shape(img)[0], np.shape(img)[1]
src = np.float32([[0,0], [0,img_width-1], [img_height-1,0], [img_height-1, img_width-1]])
dst = np.float32([p1, p2, p3, p4])

transform = cv2.getPerspectiveTransform(src, dst)


def bilinear(img):
    buf = np.zeros((3*img.shape[0], 3*img.shape[1],
                    img.shape[2]), dtype=np.uint8)
    for i in range(3*img.shape[0]):
        for j in range(3*img.shape[1]):
            SrcX=float((i+0.5)/3-0.5)
            SrcY=float((j+0.5)/3-0.5)
            if int(SrcX) >= img.shape[0]-2 and int(SrcY) >= img.shape[1]-2:
                buf[i,j,:]=img[int(SrcX), int(SrcY), :]            
            elif int(SrcX) >= img.shape[0]-2:
                f_1 = img[int(SrcX), int(SrcY), :] 
                f_2 = img[int(SrcX),int(SrcY)+1, :]
                buf[i, j, :] = (int(SrcY)+1-SrcY)*f_1+(SrcY-int(SrcY))*f_2 
            elif int(SrcY) >= img.shape[1]-2:
                buf[i, j, :] = (int(SrcX)+1-SrcX)*img[int(SrcX), int(SrcY), :] + \
                (SrcX-int(SrcX))*img[int(SrcX)+1, int(SrcY), :]
            else:
                f_1 = (int(SrcX)+1-SrcX)*img[int(SrcX), int(SrcY), :] + \
                    (SrcX-int(SrcX))*img[int(SrcX)+1, int(SrcY), :]
                f_2 = (int(SrcX)+1-SrcX)*img[int(SrcX),int(SrcY)+1, :] + \
                    (SrcX-int(SrcX))*img[int(SrcX)+1,int(SrcY)+1 , :]
                buf[i, j, :] = (int(SrcY)+1-SrcY)*f_1+(SrcY-int(SrcY))*f_2
    return buf


def project(image, video):
    
    buf = np.zeros(image.shape)
    invers=np.linalg.inv(transform)
    it = np.nditer(connect, flags=['multi_index'], op_flags=['readwrite'])
    for x in it:

        tran = np.multiply(transform, multi_index)
        if tran

    print(transform)



if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    img = cv2.imread('warp.jpg',cv2.IMREAD_COLOR)  
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            break
        cv2.imshow("actual", frame)
        key = cv2.waitKey(1)
        if key  == ord("q"):

            print("capture")
        elif key == ord("b"):
            break
        tranform = project(img, frame)

    cv2.imshow('test',img)

    cv2.waitKey(0)  
    cv2.destroyAllWindows()