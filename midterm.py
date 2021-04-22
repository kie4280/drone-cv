from Tello_Video import tello
import cv2
import time
from Tello_Video import calibrate
import numpy as np
import math
from HUD import HUD
import multiprocessing as mp
import VideoProcessing
import traceback


def center(drone, pos_in, rot_in, threshold_xyz=(10, 10, 10),
           offset_z=100, offset_xy=(0, 0), threshold_rot=5):
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
    speed = 10 if abs(x) > 0 or abs(y) > 0 or abs(z) > 0 else 20
    drone.set_speed(speed)
    if x == 0 and y == 0 and z == 0:
        drone.hover()
    else:
        if x < 0:
            drone.move_left(-x)
        else:
            drone.move_right(x)
        if y < 0:
            drone.move_up(-y)
        else:
            drone.move_down(y)

        if z < 0:
            drone.move_backward(-z)
        else:
            drone.move_forward(z)

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
    center(drone, ids[0]["tvec"], ids[0]["rvec"], offset_z=80)


def position_1(drone, ids: map):
    if 3 not in ids or 0 in ids:
        return False
    (x, y, z) = ids[3]["tvec"]
    print(x, y, z)
    center(drone, ids[3]["tvec"], ids[3]["rvec"], offset_z=60)
    if abs(x) < 10 and abs(y) < 10 and abs(z) < 70:
        return True
    else:
        return False


def down(drone):
    print('down')
    drone.move_down(50)
    time.sleep(8)
    drone.move_forward(120)
    time.sleep(8)
    drone.move_up(50)


def position_2(drone, ids: map):
    if 4 not in ids or (0 in ids or 3 in ids):
        return False
    (x, y, z) = ids[4]["tvec"]
    print(x, y, z)
    center(drone, ids[4]["tvec"], ids[4]["rvec"], offset_z=60)
    if abs(x) < 12 and abs(y) < 12 and z < 70:
        return True
    else:
        return False


def jump(drone):
    print('jump')
    drone.move_up(100)
    time.sleep(4)
    drone.move_forward(120)
    time.sleep(4)
    drone.move_down(100)


def position_3(drone, ids: map):
    if 5 not in ids or (0 in ids or 3 in ids or 4 in ids):
        return False
    (x, y, z) = ids[5]["tvec"]
    center(drone, ids[5]["tvec"], ids[5]["rvec"], threshold_xyz=(7, 7, 7))
    if abs(x) < 10 and abs(y) < 10 and abs(z-90) < 10:
        return True
    else:
        return False


bat_last = time.time()


def show_battery(drone):
    global bat_last

    if time.now() - bat_last > 5000:
        bat_last = time.time()
        bat: int = drone.get_battery()
        hud.update("battery", bat)


value_channel = mp.Pipe()
cmd_channel = mp.Pipe()


def drone_control():
    drone = tello.Tello('', 8889, command_timeout=0.08)

    time.sleep(8)
    p = mp.Process(target=VideoProcessing.main,
                   args=[cmd_channel, value_channel])
    p.daemon = True
    p.start()
    checkpoint_1: int = 0
    checkpoint_2: int = 0
    checkpoint_3: int = 0

    while (True):
        try:
            cmd_channel[1].send(1)
            ids, key = value_channel[0].recv()
            drone.keyboard(key)

            if len(ids) == 0:
                continue
            print(key, ids)
            follow(drone, ids)
            if checkpoint_1 < 10:
                checkpoint_1 += int(position_1(drone, ids))
                if checkpoint_1 > 9:
                    down(drone)
            if checkpoint_2 < 10:
                checkpoint_2 += int(position_2(drone, ids))
                if checkpoint_2 > 9:
                    jump(drone)
            if checkpoint_3 < 15:
                checkpoint_3 += int(position_3(drone, ids))
                if checkpoint_3 > 14:
                    drone.land()

        except AssertionError as ae:
            traceback.print_exc()
            print(ae)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()
            print(e)


if __name__ == "__main__":
    drone_control()
