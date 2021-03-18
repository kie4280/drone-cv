from functools import total_ordering
import cv2
import numpy as np

def statistics(img):
    
    shape = np.shape(img)
    total_pixel = shape[0]*shape[1]
    buf = np.asfarray(tally(img)) / total_pixel
    return buf

def tally(img):
    buf = np.zeros((256), dtype=np.int)
    shape = np.shape(img)
    total_pixel = shape[0]*shape[1]
    for i in np.reshape(img, (total_pixel)):
        buf[i] += 1
    return buf


def histogram(img):
    trans = np.zeros(256, dtype=np.uint8)
    stat = statistics(img)
    for i in range(1, 256):
        stat[i] = stat[i-1] + stat[i]
        trans[i] = round(stat[i]*255)
    shape = np.shape(img)
    total_pixel = shape[0]*shape[1]
    img = np.reshape(img, (total_pixel))
    for i in range(shape[0]*shape[1]):
        img[i]=trans[img[i]]
    img = np.reshape(img, shape)
    return img

 

def threshold_2(img):
    stat = statistics(img)
    n_o = 0
    mean_b = 0
    n_b = 0
    max = 0 
    total = 0
    for i in range(256):
        n_o += stat [i]
        total += stat [i] * i
    mean_o = total /n_o
    for i in range (0,255):
        n_b_original = n_b
        n_o_original = n_o
        n_b += stat [i]
        n_o -= stat [i]
        mean_b = (mean_b*n_b_original +stat[i]*i)/n_b
        mean_o = (mean_o*n_o_original -stat[i]*i)/n_o
        variance=n_b*n_o*(mean_b-mean_o)**2
        if variance >max:
            max=variance
            thres = i
    shape = np.shape(img)
    total_pixel = shape[0]*shape[1]
    img = np.reshape(img, (total_pixel))
    for i in range(shape[0]*shape[1]):
        if img [i] >= thres:
            img[i]=255
        else:
            img[i]=0
    img = np.reshape(img, shape)
    print(thres)
    return img

def threshold_3(img):
    stat = statistics(img)
    for i in range(256):
        stat[i] = stat[i-1] + stat[i]
    mu_b = 0; mu_o = 0; T_ = 0
    max_between = 0
    fg = np.where(img >= 1, img, 0)
    mu_b = np.sum(fg) / np.count_nonzero(fg)  
    bg = np.where(img == 0, img, 0)
    mu_o = np.sum(bg) / np.count_nonzero(bg)
    for T in range(1, 256):
        n_b = stat[T-1]
        n_o = stat[T] - n_b
        max_temp = n_b * n_o *(mu_b-mu_o)**2
        if max_temp > max_between:
            max_between = max_temp  
            T_ = T      
               
    return np.where(img > T_, 255, 0)
if __name__ == "__main__":
    MJ_img = cv2.imread("lab2/mj.tif")[:,:,0]
    
    input_img = cv2.imread("lab2/input.jpg")[:,:,0]
    out_img = histogram(MJ_img)
    after_otsu= threshold_3(input_img)
    cv2.imshow("ddd",after_otsu)
    cv2.imshow("sdf", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()