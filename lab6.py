from Tello_Video import tello
import cv2
import time
from Tello_Video import calibrate
import threading
import math
import numpy as np

cal = calibrate.Calibrate()
k0=100
Threshold_x=0.05
Threshold_y=0.05
Threshold_z=0.05
Threshold_rotate=5


def run_control(drone):
    while True:
        key = cv2.waitKey(1)
        drone.keyboard(key)


def follow_aruco():
    drone = tello.Tello('', 8889)
    time.sleep(10)
    #cap = cv2.VideoCapture(1)
    (intrinsic, distortion) = cal.load_calibrate_file("drone.xml")
    while (True):
        key=cv2.waitKey(1)
        drone.keyboard(key)
        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("drone", frame)
        
        
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Initialize the detector parameters using default values
        parameters = cv2.aruco.DetectorParameters_create()
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary,
                                                                                parameters=parameters)
        if len(markerCorners) == 0:
            continue
        frame = cv2.aruco.drawDetectedMarkers(
            frame, markerCorners, markerIds)
        # Pose estimation for single markers.
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners,
                                                                        15, intrinsic, distortion)
        if rvec.any() == None: 
            continue
        frame = cv2.aruco.drawAxis(
            frame, intrinsic, distortion, rvec, tvec, 2)

        frame = cv2.putText(frame, 'z'+str(tvec[0][0][2]), (int(frame.shape[0]/2) + int(tvec[0][0][0]), int(frame.shape[1]/2) + int(tvec[0][0][1])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('drone', frame)
        cv2.waitKey(1)
        (x,y,z)=tvec[0][0]/k0
        if x < -Threshold_x :
            drone.move_left(-x)  
        if x > Threshold_x:
            drone.move_right(x)
        if y < -Threshold_y :
            drone.move_up(-y)  
        if y > Threshold_y:
            drone.move_down(y)
        z=(z-1)/2
        if z > Threshold_z :
            drone.move_forward(z)
            # print(str(z)+' forward ')
        if z < -Threshold_z:
            drone.move_backward(-z)
            # print(str(-z)+' backward ')
        rvec_matrix = cv2.Rodrigues(rvec)
        proj_z=np.matmul(rvec_matrix[0],np.array([0,0,1]).T)
        rad=math.atan2(proj_z[0], proj_z[2])
        degree=np.rad2deg(rad)
        degree=(degree-180)%360
        if degree>180:
            degree-=360
        if degree >Threshold_rotate:
            drone.rotate_cw(degree)
            print(" clockwisw")
        if degree <-Threshold_rotate:
            drone.rotate_ccw(-degree)
            print("counter clockwisw")


def video_thread(drone):
    (intrinsic, distortion) = cal.load_calibrate_file("drone.xml")
    while (True):
        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(1)

        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Initialize the detector parameters using default values
        parameters = cv2.aruco.DetectorParameters_create()
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary,
                                                                               parameters=parameters)
        if len(markerCorners) == 0:
            continue
        frame = cv2.aruco.drawDetectedMarkers(
            frame, markerCorners, markerIds)
        # Pose estimation for single markers.
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners,
                                                                     15, intrinsic, distortion)

        frame = cv2.aruco.drawAxis(
            frame, intrinsic, distortion, rvec, tvec, 2)

        print(tvec)
        # frame = cv2.putText(frame, 'z'+str(tvec[0][0][2]), (int(frame.shape[0]/2) + int(tvec[0][0][0]), int(frame.shape[1]/2) + int(tvec[0][0][1])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        #                     1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('drone', frame)
        # drone.keyboard(key)


def main():
    drone = tello.Tello('', 8889)
    time.sleep(5)
    ct = threading.Thread(target=run_control,
                          name="control thread", args=[drone])
    ct.start()

    vt = threading.Thread(target=video_thread,
                          name="video thread", args=[drone])
    vt.start()


if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    # calibrate.start_calibrate()
