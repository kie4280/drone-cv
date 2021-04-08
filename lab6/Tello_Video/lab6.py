import tello
import cv2
import time
import calibrate
import threading

cal = calibrate.Calibrate()


def track_marker(drone, ):
    pass


def run_control(drone):
    while True:
        key = cv2.waitKey(1)
        drone.keyboard(key)


def video_thread(drone):
    (intrinsic, distortion) = cal.load_calibrate_file("")
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
                          name="control thread", args=(drone))
    ct.start()

    vt = threading.Thread(target=video_thread,
                          name="video thread", args=(drone))
    vt.start()


if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    # calibrate.start_calibrate()
