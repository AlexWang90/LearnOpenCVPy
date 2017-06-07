"""
Created on 20170607
author:Alex Wang
"""
import numpy as np
import cv2

def test():
    img = cv2.imread('E://workspace/temp/test.jpg',0)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()