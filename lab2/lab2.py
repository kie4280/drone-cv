from functools import total_ordering
import cv2
import numpy as np

def statistics(img):
    buf = np.zeros((256), dtype=np.float)
    shape = np.shape(img)
    total_pixel = shape[0]*shape[1]
    for i in np.reshape(img, (total_pixel)):
        buf[i] += 1
    buf = buf / total_pixel
    return buf



def histogram(img):
    trans = np.zeros(256, dtype=np.uint8)
    stat = statistics(img)
    for i in range(1, 255):
        stat[i] = stat[i-1] + stat[i]
        trans[i] = round(stat[i]*255)
    shape = np.shape(img)
    total_pixel = shape[0]*shape[1]
    img = np.reshape(img, (total_pixel))
    for i in range(shape[0]*shape[1]):
        img[i]=trans[img[i]]
    img = np.reshape(img, shape)
    return img


if __name__ == "__main__":
    MJ_img = cv2.imread("lab2/mj.tif")[:,:,0]
    
    input_img = cv2.imread("lab2/input.jpg")
    out_img = histogram(MJ_img)
    cv2.imshow("sdf", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()