from Tello_Video import tello
import cv2
import time
from Tello_Video import calibrate
import numpy as np
import math


cal = calibrate.Calibrate()
(intrinsic, distortion) = cal.load_calibrate_file("drone.xml")

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()
# Detect the markers in the image


def center(drone, pos_in, rot_in, threshold_xyz=(10, 10, 10),
           offset_z=100, offset_xy=(0, 0), threshold_rot=1):
    x, y, z = pos_in
    x = x - offset_xy[0]
    y = y - offset_xy[1]
    z = z - offset_z
    
    if abs(x) < threshold_xyz[0]:
        x = 0
    elif abs(x) < 20:
        x = 20 if x > 0 else -20
    if abs(y) < threshold_xyz[1]:
        y = 0
    elif abs(y) < 20:
        y = 20 if y > 0 else -20
    if abs(z) < threshold_xyz[2]:
        z = 0
    elif abs(z) < 20:
        z = 20 if z > 0 else -20
    speed = 10 if abs(x) > 0 or abs(y) >0 or abs(z) > 0 else 20
    if x == 0 and y == 0 and z == 0:
        drone.hover()
    else:
        drone.go(x, y, z, speed)

    rvec_matrix = cv2.Rodrigues(rot_in)
    proj_z = np.matmul(rvec_matrix[0], np.array([0, 0, 1]).T)
    rad = math.atan2(proj_z[0], proj_z[2])
    degree = np.rad2deg(rad)
    degree = (degree-180) % 360
    if degree > 180:
        degree -= 360
    if degree > threshold_rot:
        drone.rotate_cw(degree)
    if degree < -threshold_rot:
        drone.rotate_ccw(-degree)


def follow(drone, ids: map):
    if 0 not in ids:
        return
    center(drone, ids[0]["tvec"], ids[0]["rvec"])


def position_1(drone, ids: map):
    if 3 not in ids or 0 in ids:
        return False
    center(drone, ids[3]["tvec"], ids[3]["rvec"])

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
    (x, y, z) = ids[4]["tvec"]
    center(drone, ids[4]["tvec"], ids[4]["rvec"])

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
    (x, y, z) = ids[5]["tvec"]
    center(drone, ids[5]["tvec"], ids[5]["rvec"])
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
    checkpoint_1: bool = False
    checkpoint_2: bool = False
    checkpoint_3: bool = False

    drone.set_speed(20)

    while (True):
        try:
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
                frame = cv2.putText(frame, 'z'+str(ids[i]["tvec"][2]),
                                    (int(frame.shape[0]/2) + int(ids[i]["tvec"][0]),
                                     int(frame.shape[1]/2) + int(ids[i]["tvec"][1])),
                                    cv2.Font_HERSHEY_DUPLEX,
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
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
        except AssertionError as ae:
            print(ae)


if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    # calibrate.start_calibrate()
    cv2.destroyAllWindows()
