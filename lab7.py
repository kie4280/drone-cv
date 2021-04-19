from Tello_Video import tello
import cv2
import time
from Tello_Video import calibrate
import numpy as np
import math


cal = calibrate.Calibrate()
(intrinsic, distortion) = cal.load_calibrate_file("drone.xml")
k0 = 100
Threshold_x = 0.02
Threshold_y = 0.02
Threshold_z = 0.02
Threshold_rotate = 2

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()
# Detect the markers in the image


def follow(drone, ids: map):
    if 0 not in ids:
        return
    (x, y, z) = ids[0]["tvec"]/k0
    if x < -Threshold_x:
        drone.move_left(-x)
    else:
        drone.move_left(20)
    if x > Threshold_x:
        drone.move_right(x)
    else:
        drone.move_right(20)
    if y < -Threshold_y:
        drone.move_up(-y)
    else:
        drone.move_up(20)
    if y > Threshold_y:
        drone.move_down(y)
    else:
        drone.move_down(20)
    z = (z-1)/1.1
    #print("aaaaa",x,y,z)
    if z > Threshold_z:
        drone.move_forward(z)
    else:
        drone.move_forward(20)
        # print(str(z)+' forward ')
    if z < -Threshold_z:
        drone.move_backward(-z)
    else:
        drone.move_backward(20)
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
    z = (z-0.7)
    if z > Threshold_z:
        drone.move_forward(z)
        # print(str(z)+' forward ')
    if z < -Threshold_z:
        drone.move_backward(-z)
        # print(str(-z)+' backward ')
    if  abs(z)<Threshold_z*20:
        print("true")
        return True
        
    return False
def down(drone):
    print('down')
    drone.move_down(0.3)
    time.sleep(4)
    drone.move_forward(1.2)
    time.sleep(4)
    drone.move_up(0.3)

def position_2(drone, ids: map):
    if 4 not in ids or (0 in ids or 3 in ids):
        return False
    (x, y, z) = ids[4]["tvec"]/k0
    
    if x < -Threshold_x:
        drone.move_left(-x)
    if x > Threshold_x:
        drone.move_right(x)
    if y < -Threshold_y:
        drone.move_up(-y)
    if y > Threshold_y:
        drone.move_down(y)
    z = (z-0.7)
    if z > Threshold_z:
        drone.move_forward(z)
        # print(str(z)+' forward ')
    if z < -Threshold_z:
        drone.move_backward(-z)
        # print(str(-z)+' backward ')
    if  abs(z)<Threshold_z*20:
        print("true")
        return True
        
    return False
def jump(drone):
    print('jump')
    drone.move_up(1)
    time.sleep(4)
    drone.move_forward(1.2)
    time.sleep(4)
    drone.move_down(1)

def position_3(drone, ids: map):
    if 5 not in ids or (0 in ids or 3 in ids or 4 in ids):
        return False
    (x, y, z) = ids[5]["tvec"]/k0
    
    if x < -Threshold_x:
        drone.move_left(-x)
    if x > Threshold_x:
        drone.move_right(x)
    if y < -Threshold_y:
        drone.move_up(-y)
    if y > Threshold_y:
        drone.move_down(y)
    z = (z-0.7)
    if z > Threshold_z:
        drone.move_forward(z)
        # print(str(z)+' forward ')
    if z < -Threshold_z:
        drone.move_backward(-z)
        # print(str(-z)+' backward ')
    rvec_matrix = cv2.Rodrigues(ids[5]["rvec"])
    proj_z = np.matmul(rvec_matrix[5], np.array([0, 0, 1]).T)
    rad = math.atan2(proj_z[0], proj_z[2])
    degree = np.rad2deg(rad)
    degree = (degree-180) % 360
    if degree > 180:
        degree -= 360
    if degree > Threshold_rotate:
        drone.rotate_cw(degree)
    if degree < -Threshold_rotate:
        drone.rotate_ccw(-degree)
    if  abs(z)<Threshold_z*20 and abs(x)<Threshold_x*10 and abs(degree)<Threshold_rotate:
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
            #print(markerIds[i], tvec)
        except AssertionError as ae:
            return frame, []

    return frame, ids


def main():
    drone = tello.Tello('', 8889)
    time.sleep(8)
    # cap = cv2.VideoCapture(1)
    checkpoint_1:bool = False
    checkpoint_2:bool = False
    checkpoint_3:bool = False

    drone.set_speed(10)
    while (True):
        key = cv2.waitKey(1)
        drone.keyboard(key)
        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("drone", frame)

        frame, ids = detect_code(frame)
        if len(ids) == 0:
            continue
        for i in ids.keys():
            frame = cv2.aruco.drawAxis(
                frame, intrinsic, distortion, ids[i]["rvec"], ids[i]["tvec"], 2)
            frame = cv2.putText(frame, 'z'+str(ids[i]["tvec"][2]), (int(frame.shape[0]/2) + int(ids[i]["tvec"][0]), int(frame.shape[1]/2) + int(ids[i]["tvec"][1])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('drone', frame)
        cv2.waitKey(1)
        follow(drone, ids)
        if not checkpoint_1:
            checkpoint_1 = position_1(drone, ids)
            if checkpoint_1:
                down(drone)
        if not checkpoint_2:
            checkpoint_2 = position_2(drone, ids)
            if checkpoint_2:
                jump(drone)
        if not checkpoint_3:
            checkpoint_3 = position_3(drone, ids)
            if checkpoint_3:
                drone.land()
        print(checkpoint_1)
        

if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    # calibrate.start_calibrate()
    cv2.destroyAllWindows()
