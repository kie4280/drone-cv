import cv2
import numpy as np
import os
import glob
from Tello_Video import tello


class Calibrate:

    def __init__(self):
        # Defining the dimensions of checkerboard
        self.CHECKERBOARD = (6, 9)
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpoints = []

        # Defining the world coordinates for 3D points
        self.objp = np.zeros(
            (1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0],
                                       0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        self.images = []
        pass

    def add_image(self, calibrate_img):
        self.images.append(calibrate_img)
        pass

    def write_calibrate_file(self, filename="1.xml"):
        self._get_cali_results()
        f = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        f.write("intrinsic", self.mtx)
        f.write("distortion", self.dist)
        f.release()

    def load_calibrate_file(self, filename="1.xml"):
        f = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.mtx = f.getNode("intrinsic").mat()
        self.dist = f.getNode("distortion").mat()

        return self.mtx, self.dist

    def _get_cali_results(self):
        if len(self.images) < 3:
            raise RuntimeError("Too small")
        for img in self.images:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(
                gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                self.objpoints.append(self.objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria)

                self.imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(
                    img, self.CHECKERBOARD, corners2, ret)
        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        return self.mtx, self.dist



def start_calibrate():
    drone = tello.Tello('', 8889)

    cal = Calibrate()
    while True:
        frame = drone.read()
        cv2.imshow("drone", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cal.add_image(frame)
        elif key == ord('e'):
            cal.write_calibrate_file("lab6/webcam.xml")
    
    cal.load_calibrate_file("lab6/webcam.xml")
