import cv2
import numpy as np
import timeit
import glob
import time

SURF_detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
SIFT_detector = cv2.xfeatures2d.SIFT_create()
ORB_detector = cv2.ORB_create()


def detect_SIFT(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (kps, features) = SIFT_detector.detectAndCompute(gray, None)
    #    kps = np.float32([kp.pt for kp in kps])
    cv2.drawKeypoints(image, kps, image, (0, 255, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    #    cv2.drawKeypoints(I,kps,I,(0,255,255))

    # cv2.imwrite('test1.png',image)
    return image, kps, features


def detect_SURF(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (kps, features) = SURF_detector.detectAndCompute(gray, None)
    #    kps = np.float32([kp.pt for kp in kps])
    cv2.drawKeypoints(image, kps, image, (0, 255, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    #    cv2.drawKeypoints(I,kps,I,(0,255,255))

    # cv2.imwrite('test1.png',image)
    return image, kps, features


def detect_ORB(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (kps, features) = ORB_detector.detectAndCompute(gray, None)
    #    kps = np.float32([kp.pt for kp in kps])
    cv2.drawKeypoints(image, kps, image, (0, 255, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    #    cv2.drawKeypoints(I,kps,I,(0,255,255))

    # cv2.imwrite('test1.png',image)
    return image, kps, features


def match(img1, img2, kps1, kps2, f1, f2, ratio=0.75):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(f1, f2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # another way
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # matches = bf.match(f1,f2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches[:10], None, flags=2)
    # cv2.imwrite("test_match2.jpg", img3)
    return img3


def test_single(img1, img2, filename="test.jpg"):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    o1, kps1, f1 = detect_ORB(img1)
    o2, kps2, f2 = detect_ORB(img2)
    o3 = match(o1, o2, kps1, kps2, f1, f2, ratio=0.75)
    cv2.imwrite("output/ORB_" + filename, o3)
    o1, kps1, f1 = detect_SIFT(img1)
    o2, kps2, f2 = detect_SIFT(img2)
    o3 = match(o1, o2, kps1, kps2, f1, f2, ratio=0.75)
    cv2.imwrite("output/SIFT_" + filename, o3)
    o1, kps1, f1 = detect_SURF(img1)
    o2, kps2, f2 = detect_SURF(img2)
    o3 = match(o1, o2, kps1, kps2, f1, f2, ratio=0.75)
    cv2.imwrite("output/SURF_" + filename, o3)


def test_time(img1, img2, filename="test.jpg"):
    res = dict()
    secs = 0
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    for _ in range(5):
        now = time.time()
        o1, kps1, f1 = detect_ORB(img1)
        o2, kps2, f2 = detect_ORB(img2)
        secs += time.time() - now
    res["orb"] = secs / 10
    secs = 0
    for _ in range(10):
        now = time.time()
        o3 = match(o1, o2, kps1, kps2, f1, f2, ratio=0.75)
        secs += time.time() - now
    res["match_orb"] = secs / 10
    secs = 0
# -----------------------------------------
    for _ in range(5):
        now = time.time()
        o1, kps1, f1 = detect_SIFT(img1)
        o2, kps2, f2 = detect_SIFT(img2)
        secs += time.time() - now
    res["sift"] = secs / 10
    secs = 0
    
    for _ in range(10):
        now = time.time()
        o3 = match(o1, o2, kps1, kps2, f1, f2, ratio=0.75)
        secs += time.time() - now
    res["match_sift"] = secs / 10
    secs = 0
# -----------------------------------------
    for _ in range(5):
        now = time.time()
        o1, kps1, f1 = detect_SURF(img1)
        o2, kps2, f2 = detect_SURF(img2)
        secs += time.time() - now
    res["surf"] = secs / 10
    secs = 0
   
    for _ in range(10):
        now = time.time()
        o3 = match(o1, o2, kps1, kps2, f1, f2, ratio=0.75)
        secs += time.time() - now
    res["match_surf"] = secs / 10
    secs = 0

    return res


if __name__ == "__main__":
    results = dict()
    img_stub = "lab11/test2_"
    with open("time_result.txt", "w+") as f:
        for i in range(1, 5):
            for j in range(1, 5):
                if i == j:
                    continue
                res = test_time(img_stub + str(i) + ".jpg", img_stub +
                                str(j) + ".jpg", filename="test_{}_{}.jpg".format(i, j))
                
                f.write("img{} and img{} orb:{}/match:{} sift:{}/match:{} surf:{}/match:{}\n".format(
                    i, j, res["orb"], res["match_orb"], res["sift"], res["match_sift"], res["surf"], res["match_surf"]
                ))

    cv2.destroyAllWindows()
    pass
