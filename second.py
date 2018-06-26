import cv2
import numpy as np
# from numpy import numarray as na
from matplotlib import pyplot as plt

img = cv2.imread("Geneva.tif")

frequency = np.zeros(256)
rows, cols = img.shape[:2]


def colect(image):
    alpha = 255/(image.shape[0]*image.shape[1])
    print(alpha)
    # print(frequency)
    for i in range(0, rows):
        for j in range(0, cols):
            for a in range(len(frequency)):
                if image[i][j][0] == a:
                    frequency[a] += 1
                    # print(image[i][j][0])
                    # print(frequency[a])
    print(frequency.max(), " min", frequency.min())
    pdf(frequency, rows, cols)


def pdf(fr, rw, cl):
    temp = np.zeros(256)
    for a in range(256):
        temp[a] = fr[a]/(rw*cl)
    cdf(temp)


def cdf(tmp):
    for i in range(1, 255):
        tmp[i] += tmp[i-1]
    plt.hist(tmp, 256, [0, 256])
    normalize(tmp)


def normalize(temp):
    normal = np.zeros(256)
    for i in range(len(frequency)):
        normal[i] = round(((temp[i]-1)/(rows*cols))*255)


# def final(norm):
#     for i in range(len()):


# cv2.imshow("1", img)
colect(img)
# hist = cv2.calcHist(img, [0], None, 256, [0, 255])
plt.hist(img.ravel(), 256, [0, 256])
# res = np.hstack((img,equ)) #side-by-side
cv2.waitKey(0)
cv2.destroyAllWindows()
