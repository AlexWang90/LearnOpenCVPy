"""
Created on 20170607
author:Alex Wang
"""
import numpy as np
import cv2

def test():
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print (flags)

    img = cv2.imread('E://workspace/temp/test.jpg')
    # cvtimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # reshape_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    reshape_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    reshape_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('image',reshape_img)
    print(img.shape)
    print(type(img))
    print(cv2.useOptimized())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()