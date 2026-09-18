import cv2 as cv
import numpy as np

img =cv.imread('photos/cat.jpg')
cv.imshow('Original Image',img)


def translate(img,x,y):
    transmat=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transmat,dimensions)

translated=translate(img,100,100)

cv.imshow('Translated Image',translated)
cv.waitKey(0)




