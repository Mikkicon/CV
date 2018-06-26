
# coding: utf-8

# In[5]:


import cv2 as cv
import numpy as np

def defineAverageBrightness(img):
    average=0
    hist = np.zeros((256,1),np.uint8)
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            brightness = int(img[x][y])
            average+=brightness
            hist[brightness][0]+=1
    return average/(img.shape[1]*img.shape[0]),hist

def u1(tk,hist):
    one=0
    two=0
    for i in range(tk):
        one+=(i*hist[i][0])
        two+=hist[i][0]
    return int(one/two)


def u2(tk, hist):
    one = 0
    two = 0
    while tk<hist.shape[0]:
        one += (tk * hist[tk][0])
        two += hist[tk][0]
        tk+=1
    return int(one/two)


def iterationalAlgo(averageBright,hist):
    tk = int(averageBright)
    tk1 = -1
    while averageBright<hist.shape[0]:
        tk1=int((u1(tk,hist)+u2(tk,hist))/2)
        if(tk==tk1):
            return tk
        tk=tk1
        tk1=0
        averageBright+=1
    return averageBright

def convertImg(img,threshold):
    res=np.zeros((img.shape[0],img.shape[1]),np.uint8)
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            if(img[x][y]>=threshold):
                res[x][y]=255
    return res

default = cv.imread('/Users/mihailpetrenko/PycharmProjects/Opencv/img/rubbish.jpg',0)
default=cv.resize(default,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
brightness,hist = defineAverageBrightness(default)
print(brightness)
threshold=iterationalAlgo(brightness,hist)
print(threshold)
res = convertImg(default,threshold)
cv.imshow('default',default)
cv.imshow('res',res)
cv.waitKey()
cv.destroyAllWindows()

