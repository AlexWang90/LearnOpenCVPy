"""
Created on 2017-06-10
author: Alex Wang
"""
import cv2
import numpy as np

def ext_sift_feature(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = ext_sift_from_img(gray_img)
    return  kp, des

def ext_sift_from_img(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img,None)
    return kp, des