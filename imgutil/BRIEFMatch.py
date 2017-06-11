"""
Created on 2017-06-10
author: Alex Wang
"""
import cv2
import numpy as np

def ext_brief_feature(img_path):
    img = cv2.imread(img_path)
    kp, des = ext_brief_from_img(img)
    return kp,des

def ext_brief_from_img(img):
    star = cv2.xfeatures2d.StarDetector_create()  ## CenSurE特征提取算法
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img,None)
    kp, des = brief.compute(img, kp)
    return kp, des