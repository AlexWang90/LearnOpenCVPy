"""
Created on 2017-06-11
author: Alex Wang
"""
import cv2
import numpy as np

def ext_orb_feature(img_path):
    img = cv2.imread(img_path)
    kp, des = ext_orb_from_img(img)

def ext_orb_from_img(img, nFeatures = 500):
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    return kp, des

def brute_force_match(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    return matches