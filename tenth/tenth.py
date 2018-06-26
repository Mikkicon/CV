# import ninth as n
import cv2
import numpy as np
import math

img = cv2.imread("img/shape.jpg")
img1 = cv2.imread("img/shape1.jpg")
rows, cols = img.shape[:2]
result = np.zeros((rows,cols,3), np.uint8)
# convex = n.contour
convex = np.ones((651,2))
forward = np.array(convex.shape[0])
# forward = np.ones(convex.shape[0])
backward = np.array(convex.shape[0])
# backward =np.ones(convex.shape[0])
f_sigma = np.array(convex.shape[0])
b_sigma = np.array(convex.shape[0])
angle_dif = np.array(convex.shape[0])
curvature = np.array(convex.shape[0])


def kokos(k):
    B =0
    for i in range(k+1,convex.shape[0]):
        print("indicator")
        print("convex[i+k,0]: ",convex[i+k,0], "convex[i,0]: ", convex[i,0], "i: ", i)
        forward[i] = math.sqrt(pow((convex[i+k][0] - convex[i][0]),2)+pow((convex[i+k][1] - convex[i][1]),2))
        print("convex[i+k,0]: ",convex[i+k][0], "convex[i,0]: ", convex[i][0], "i: ", i)
        print("Forward: ", forward[i])
        print("\n")
        backward[i] = math.sqrt(pow(convex[i-k][0] - convex[i][0],2)+pow(convex[i-k][1] - convex[i][1],2))
        if (convex[i+k][1] - convex[i][1]) !=0 or (convex[i-k][1] - convex[i][1]) !=0:
            f_sigma[i] = math.atan((convex[i+k][0] - convex[i][0])/(convex[i+k][1] - convex[i][1]))
            b_sigma[i] = math.atan((convex[i-k][0] - convex[i][0])/(convex[i-k][1] - convex[i][1]))
        elif (convex[i+k][1] - convex[i][1]) ==0 or (convex[i-k][1] - convex[i][1]) ==0:
            f_sigma[i] = math.atan((convex[i+k][1] - convex[i][1])/(convex[i+k][0] - convex[i][0]))
            b_sigma[i] = math.atan((convex[i-k][1] - convex[i][1])/(convex[i-k][0] - convex[i][0]))
        else:
            print("Exception")
        angle_dif[i] = f_sigma[i]-((f_sigma+b_sigma)/2)
        curvature[i] = (angle_dif *(forward[i]+backward[i]))/(2*forward[i]*backward[i])
        B += pow(curvature[i],2)
    return B/convex.shape[0]

# k = input('Input length of each segment: ')
# print(convex)
answer = kokos(2)
print(answer)
k=8
print(convex.shape)