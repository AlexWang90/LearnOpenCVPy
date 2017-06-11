"""
Created on 2017-06-10
author: Alex Wang
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from imgutil import SiftMatch
from imgutil import SurfMatch
from imgutil import BRIEFMatch
from imgutil import ORBMatch

def test():
    # test_sift()
    # test_surf()
    # test_brief()
    test_orb()

def test_surf():
    img = cv2.imread("E://workspace/temp/sift_1.jpg")
    gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp,des = SurfMatch.ext_surf_from_img(gray_img1)
    print(des.shape)
    img = cv2.drawKeypoints(img, kp, None, (255,0,0),4)
    plt.subplot(121)
    plt.imshow(img)


    img2 = cv2.imread("E://workspace/temp/sift_2.jpg")
    gray_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp,des = SurfMatch.ext_surf_from_img(gray_img2)
    print(des.shape)
    img2 = cv2.drawKeypoints(img2, kp, None, (255,0,0),4)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

def test_sift():
    img = cv2.imread("E://workspace/temp/sift_1.jpg")
    gray_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    kp, des = SiftMatch.ext_sift_from_img(gray_img)
    print(des.shape)
    dummy = np.zeros((1,1))
    img=cv2.drawKeypoints(img,kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("sift ext done")
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    img2 = cv2.imread("E://workspace/temp/sift_2.jpg")
    gray_img2 = cv2.cvtColor(img2,  cv2.COLOR_BGR2GRAY)
    kp, des = SiftMatch.ext_sift_from_img(gray_img2)
    img2=cv2.drawKeypoints(img2,kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2)
    plt.show()

def test_brief():
    img = cv2.imread("E://workspace/temp/sift_1.jpg")
    kp, des = BRIEFMatch.ext_brief_from_img(img)
    print(des.shape)
    # print(des)
    img = cv2.drawKeypoints(img, kp, None, (255,0,0),4)
    plt.imshow(img)
    plt.show()

def test_orb():
    img = cv2.imread("E://workspace/temp/sift_1.jpg")
    kp, des = ORBMatch.ext_orb_from_img(img)
    print(des.shape)
    img = cv2.drawKeypoints(img, kp, None, (255,0,0),4)
    plt.imshow(img)
    plt.show()

def load_gray_img(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img