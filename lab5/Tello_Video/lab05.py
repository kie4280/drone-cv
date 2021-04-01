import tello
import cv2
import time
import calibrate

cal = calibrate.Calibrate()


def main():
    drone = tello.Tello('', 8889)
    time.sleep(15)
    #cap = cv2.VideoCapture(1)
    intrinsic, distortion = None, None
    while (True):
        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(1)

        if key != -1:
            if key == ord('q'):
                print("capture")
                cal.add_image(frame)
            elif key == ord('z'):
                cal.write_calibrate_file()
            elif key == ord('/'):
                (intrinsic, distortion) = cal.get_cali_results()
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
                frame = cv2.putText(frame, 'z'+str(tvec[0][0][2]), (int(frame.shape[0]/2) + int(tvec[0][0][0]), int(frame.shape[1]/2) + int(tvec[0][0][1])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                    1, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('aaa', frame)
                #drone.keyboard(key)


if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    cv2.destroyAllWindows()
