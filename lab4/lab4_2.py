import cv2 
import numpy as np

p1 = (202, 242)
p2 = (241, 464)
p3 = (631, 255)
p4 = (765, 432)

def project(image, video,transform):
    video_shape=(np.shape(video)[0],np.shape(video)[1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cur_pos=(i,j,1)
            response=np.matmul(transform,cur_pos)
            response_x=response[0]/response[2]
            response_y=response[1]/response[2]
            #print(np.matmul(invers,(243,462,1)))
            if (response_x>=0 and response_x<= video.shape[0]) and (response_y>=0 and response_y<= video.shape[1]):
                if int(response_x) >= video_shape[0]-1 and int(response_y) >= video_shape[1]-1:
                    image[i,j,:]=video[int(response_x), int(response_y), :]            
                elif int(response_x) >= video_shape[0]-1:
                    f_1 = video[int(response_x), int(response_y), :] 
                    f_2 = video[int(response_x),int(response_y)+1, :]
                    image[i,j,:]= (int(response_y)+1-response_y)*f_1+(response_y-int(response_y))*f_2 
                elif int(response_y) >= video_shape[1]-1:
                    image[i,j,:]= (int(response_x)+1-response_x)*video[int(response_x), int(response_y), :] + \
                    (response_x-int(response_x))*video[int(response_x)+1, int(response_y), :]
                else:
                    f_1 = (int(response_x)+1-response_x)*video[int(response_x), int(response_y), :] + \
                        (response_x-int(response_x))*video[int(response_x)+1, int(response_y), :]
                    f_2 = (int(response_x)+1-response_x)*video[int(response_x),int(response_y)+1, :] + \
                        (response_x-int(response_x))*video[int(response_x)+1,int(response_y)+1 , :]
                    image[i,j,:] = (int(response_y)+1-response_y)*f_1+(response_y-int(response_y))*f_2
    return image



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    img = cv2.imread('warp.jpg',cv2.IMREAD_COLOR)
    img_height, img_width= np.shape(img)[0], np.shape(img)[1]
    dst = np.float32([p1, p2, p3, p4])  
    transform = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imshow("actual", frame)
        src_height,src_width = np.shape(frame)[0] , np.shape(frame)[1]
        src = np.float32([(0,0),  (0,src_width-1), (src_height-1,0), (src_height-1, src_width-1)])
        if transform is None:
            transform = cv2.getPerspectiveTransform(dst, src)
        key = cv2.waitKey(1)
        
        result = project(img,frame,transform)
        if key == ord("b"):
            break
        
        cv2.imshow("org",result)
        cv2.waitKey(1)
        #print(np.matmul(transform,(0,src_width-1,0)),'  ' ,np.matmul(transform,(src_height-1, src_width-1,1)))
        #print(transform)

    cv2.destroyAllWindows()
