import cv2
import numpy as np

def rotate(img):
    height = img.shape[1]
    width = img.shape[0]
    buf = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    for i in range(height):
        buf[i, :, :] = img[:, height-i-1, :]
    return buf


def flip(img):
    buf = np.zeros(img.shape, dtype=np.uint8)
    width = img.shape[1]
    for i in range(width):
        buf[:, i, :] = img[:, width-i-1, :]
    return buf


def nearest(img):
    buf = np.zeros((3*img.shape[0], 3*img.shape[1],
                    img.shape[2]), dtype=np.uint8)
    for i in range(3*img.shape[0]):
        for j in range(3*img.shape[1]):
            buf[i, j, :] = img[i//3, j//3, :]
    return buf


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


if __name__ == "__main__":
    kobe_img = cv2.imread("lab1/kobe.jpg")
    IU_img = cv2.imread("lab1/IU.png")
    curry_img = cv2.imread("lab1/curry.jpg")
    out = flip(kobe_img)    
    cv2.imwrite('lab1/flip.png',out)    
    out = rotate(curry_img)
    cv2.imwrite('lab1/rotate.png',out)    
    out = bilinear(IU_img)
    cv2.imwrite('lab1/bilinear.png',out)
    out = nearest(IU_img)
    cv2.imwrite('lab1/nearest.png', out)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
