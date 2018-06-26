import cv2 
import numpy as np
import math 

img = "1.jpg"
default = cv2.imread(img)

def count(kernel1,kernel2):
    result=0
    for i in range(3):
        for j in range(3):
            result+=kernel1[j][i]*kernel2[j][i]
    return result


def sobel(imgU):
    default = cv2.imread(imgU)
    default = cv2.resize(default, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(default, cv2.COLOR_BGR2GRAY)

    kernel1 = [[-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1]]
    kernel2 = [[-1, -2, -1], 
                [0, 0, 0], 
                [1, 2, 1]]

    res1 = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), np.uint8)
    res2 = np.zeros((gray_image.shape[0], gray_image.shape[1], 1), np.uint8)
    for y in range(gray_image.shape[1] - 3):
        for x in range(gray_image.shape[0] - 3):
            kernel0 = gray_image[x:x + 3, y:y + 3]
            pixelX = count(kernel1, kernel0)
            pixelY = count(kernel2, kernel0)
            res1[x][y] = math.sqrt(math.pow(pixelX, 2) + math.pow(pixelY, 2))

            if(pixelX<0):
                pixelX*=-1
            if (pixelY < 0):
                pixelY *= -1
            if(pixelX!=0):
                res2[x][y] = math.atan(pixelY/pixelX)
    return res1,res2


res,res1 = sobel(img)
#cv2.imshow("default",default)
cv2.imshow("sobel",res)
cv2.imshow("default",default)
cv2.waitKey(0)
cv2.destroyAllWindows()