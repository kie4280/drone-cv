import multiprocessing as mp
import threading
from Tello_Video import tello
import cv2
import time
from Tello_Video import calibrate
import numpy as np
from HUD import HUD
import traceback


cal = calibrate.Calibrate()
(intrinsic, distortion) = cal.load_calibrate_file("1.xml")

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()
# Detect the markers in the image


hud = HUD()
hud.addFields("battery", "cmd")


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


bat_last = time.time()

video_frame = np.zeros(shape=(720, 720, 3), dtype=np.uint8)


def _receive_video_thread():
    global video_frame
    video_ip = "udp://{}:{}".format("0.0.0.0", 11111)
    video_capture = cv2.VideoCapture(video_ip)
    retval, video_frame = video_capture.read()
    while retval:
        retval, video_frame = video_capture.read()
        video_frame = video_frame[..., ::-1]  # From BGR to RGB


key: int = -1
ids: map = {}
value_channel: mp.Pipe()
cmd_channel: mp.Pipe()


def main(_cmd, _val):
    global video_frame, key, ids, cmd_channel, value_channel
    cmd_channel = _cmd
    value_channel = _val
    vt = threading.Thread(target=_receive_video_thread)
    vt.daemon = True
    vt.start()

    # cap = cv2.VideoCapture(0)
    # drone.set_speed(20)
    while (True):

        try:
            key = cv2.waitKey(1)
            request_data = cmd_channel[0].poll()
            if request_data:
                cmd, data = cmd_channel[0].recv()
                if cmd == 1:
                    value_channel[1].send((ids, key))
                elif cmd == 2:
                    for i in data.keys():
                        hud.update(i, data[i])
                    value_channel[1].send(0)
            # ret, video_frame = cap.read()
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
            video_frame, ids = detect_code(video_frame)
            video_frame = hud.getFrame(video_frame)
            if len(ids) == 0:
                cv2.imshow("drone", video_frame)
                continue
            for i in ids.keys():
                video_frame = cv2.aruco.drawAxis(
                    video_frame, intrinsic, distortion, ids[i]["rvec"], ids[i]["tvec"], 2)

            cv2.imshow('drone', video_frame)

        except AssertionError as ae:
            traceback.print_exc()
            print(ae)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()
            print(e)


if __name__ == "__main__":
    main()
    # Load the predefined dictionary
    # calibrate.start_calibrate()
    cv2.destroyAllWindows()
