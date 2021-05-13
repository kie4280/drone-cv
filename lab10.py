from Tello_Video import tello, calibrate
import cv2
import time
import numpy as np
import math
cal = calibrate.Calibrate()
(intrinsic, distortion) = cal.load_calibrate_file("drone.xml")
k0 = 120
Threshold_x = 0.02
Threshold_y = 0.05
Threshold_z = 0.01
Threshold_rotate = 10

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()
# Detect the markers in the image


def follow(drone, ids: map):
    if 0 not in ids:
        return
    (x, y, z) = ids[0]["tvec"]/k0

    # print(str(-z)+' backward ')
    rvec_matrix = cv2.Rodrigues(ids[0]["rvec"])
    proj_z = np.matmul(rvec_matrix[0], np.array([0, 0, 1]).T)
    rad = math.atan2(proj_z[0], proj_z[2])
    degree = np.rad2deg(rad)
    degree = (degree-180) % 360
    if degree > 180:
        degree -= 360
    if degree > Threshold_rotate:
        drone.rotate_cw(degree)
    if degree < -Threshold_rotate:
        drone.rotate_ccw(-degree)

    if x < -Threshold_x:
        drone.move_left(-x)
    if x > Threshold_x:
        drone.move_right(x)
    if y < -Threshold_y:
        drone.move_up(-y)
    if y > Threshold_y:
        drone.move_down(y)
    z = (z-1)/1.1
    # print("aaaaa",x,y,z)
    if z > Threshold_z:
        drone.move_forward(z)
        # print(str(z)+' forward ')
    if z < -Threshold_z:
        drone.move_backward(-z)


def position_1(drone, ids: map):
    if 3 not in ids or 0 in ids:
        return False
    (x, y, z) = ids[3]["tvec"]/k0

    if x < -Threshold_x:
        drone.move_left(-x)
    if x > Threshold_x:
        drone.move_right(x)
    if y < -Threshold_y:
        drone.move_up(-y)
    if y > Threshold_y:
        drone.move_down(y)
    z = (z-0.6)
    if z > Threshold_z:
        drone.move_forward(z)
        # print(str(z)+' forward ')
    if z < -Threshold_z:
        drone.move_backward(-z)
        # print(str(-z)+' backward ')
    if abs(z) < Threshold_z*20:
        print("true")
        return True

    return False


def detect_code(frame):
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary,
                                                                           parameters=parameters)
    # print(markerCorners)
    if len(markerCorners) == 0:
        return frame, []
    frame = cv2.aruco.drawDetectedMarkers(
        frame, markerCorners, markerIds)
    # Pose estimation for single markers.

    ids = {}

    for i in range(len(markerIds)):
        try:
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i],
                                                                         15, intrinsic, distortion)
            ids[int(markerIds[i])] = {"tvec": tvec[0][0], "rvec": rvec[0][0]}
        except AssertionError as ae:
            print(ae)
        # print(markerIds[i], tvec)

    return frame, ids


def blue_line_filter(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    cv2.imshow("original", frame)
    lower_green = np.array([0, 50, 50])
    upper_green = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    return mask


def main():
    drone = tello.Tello('', 8889)
    time.sleep(15)
    # cap = cv2.VideoCapture(0)
    cur_dir_LR = 0
    cur_dir_UD = 0
    checkpoint_1: bool = False
    cur_dir_LR = 1

    while (True):
        # ret, frame = cap.read()
        # if ret == False:
        #     break
        key = cv2.waitKey(1)
        drone.keyboard(key)
        frame = drone.read()
        buf = np.zeros(frame.shape, dtype=np.uint8)
        mask = blue_line_filter(frame)
        buf[:, :, 1] = mask[:, :]

        cv2.imshow("mask", buf)

        squares = []
        frame_size = (frame.shape[0] // 2, frame.shape[1] // 2)
        # print(frame_size)
        width_center = frame.shape[1] // 3
        height_center = frame.shape[0] // 3
        pixel_tot = width_center * height_center
        left = np.sum(mask[height_center:height_center *
                           2, :width_center]) / pixel_tot
        right = np.sum(mask[height_center:height_center *
                            2, width_center*2:]) / pixel_tot
        top = np.sum(
            mask[:height_center, width_center:width_center*2]) / pixel_tot
        bottom = np.sum(
            mask[height_center*2:, width_center:width_center*2]) / pixel_tot

        boxes = (right > 30, left > 30, top > 30, bottom > 30)
        print(boxes)
        if not boxes[0] and boxes[1] and cur_dir_LR == 1:
            cur_dir_LR = 0
            if boxes[2] and not boxes[3]:
                cur_dir_UD = 1
            elif not boxes[2] and boxes[3]:
                cur_dir_UD = -1          
        
        elif boxes[0] and not boxes[1] and cur_dir_LR == -1:
            cur_dir_LR = 0
            if boxes[2] and not boxes[3]:
                cur_dir_UD = 1
            elif not boxes[2] and boxes[3]:
                cur_dir_UD = -1 

        if not boxes[2] and boxes[3] and cur_dir_UD == 1:
            cur_dir_UD = 0
            if boxes[0] and not boxes[1]:
                cur_dir_LR = 1
            elif not boxes[0] and boxes[1]:
                cur_dir_LR = -1

        elif boxes[2] and not boxes[3] and cur_dir_UD == -1:
            cur_dir_UD = 0
            if boxes[0] and not boxes[1]:
                cur_dir_LR = 1
            elif not boxes[0] and boxes[1]:
                cur_dir_LR = -1
            
        if cur_dir_LR > 0:
            drone.move_right(20)
        elif cur_dir_LR < 0:
            drone.move_left(20)
        
        if cur_dir_UD > 0:
            drone.move_up(20)
        elif cur_dir_UD < 0:
            drone.move_down(20)
            


        frame, ids = detect_code(frame)
        if len(ids) == 0:
            continue
        for i in ids.keys():
            frame = cv2.aruco.drawAxis(
                frame, intrinsic, distortion, ids[i]["rvec"], ids[i]["tvec"], 2)
            frame = cv2.putText(frame, 'z'+str(ids[i]["tvec"][2]), (int(frame.shape[0]/2) + int(ids[i]["tvec"][0]), int(frame.shape[1]/2) + int(ids[i]["tvec"][1])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                1, (255, 0, 0), 1, cv2.LINE_AA)

        if 4 in ids.keys():
            checkpoint_1 = True
        cv2.imshow('drone', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    # calibrate.start_calibrate()
    cv2.destroyAllWindows()
