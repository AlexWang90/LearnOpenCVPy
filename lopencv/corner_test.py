"""
Created on 2017-06-10
author:Alex Wang
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def test_harr():
    img = cv2.imread('E://workspace/temp/test.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    cornor = cv2.cornerHarris(gray_img, 2, 3, 0.04)

    cornor = cv2.dilate(cornor, None)  ##加宽
    img[cornor > 0.01 * cornor.max()] = [0, 0, 255]  ##阈值过滤
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def test_shi_tomasi():
    img = cv2.imread('E://workspace/temp/test.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img, 25, 0.01, 10)
    corners = np.int16(corners)

    for i in corners:
        x, y = i.reshape(-1)
        cv2.circle(img, (x, y), 3, 255, -1)
    plt.imshow(img)
    plt.show()


def test_fast():
    img = cv2.imread('E://workspace/temp/test.jpg')
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    dummy = np.zeros((1, 1))
    img2 = cv2.drawKeypoints(img, kp, dummy, color=(255, 0, 0))
    plt.imshow(img2)
    plt.show()


if __name__ == "__main__":
    # test_harr()
    # test_shi_tomasi()
    test_fast()
