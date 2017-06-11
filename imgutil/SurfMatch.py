"""
Created on 2017-06-10
author: Alex Wang
"""
import cv2
import numpy as np

def ext_surf_feature(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = ext_surf_from_img(gray_img)
    return  kp, des

def ext_surf_from_img(gray_img, hessianThreshold = 400, extended= False, upright = False):
    """
    :param gray_img:
    :param hessianThreshold:
    :param extended:是否扩展到128维
    :param upright:如果upright=True，不计算orientation
    :return:
    """
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(hessianThreshold) # Hessian hessianThreshold, 一般300-500
    surf.setExtended(extended)
    surf.setUpright(upright)
    kp, des = surf.detectAndCompute(gray_img,None)
    return kp, des