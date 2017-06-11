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
    """
    :param gray_img: gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    :return:
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img,None)
    return kp, des

def brute_force_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    return good_matches

def kdtree_match(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good_matches = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_matches.append([m])
    return good_matches







