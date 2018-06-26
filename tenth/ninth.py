import cv2
# import time
import numpy as np
# from matplotlib import pyplot as plt

img = cv2.imread("img/shape.jpg")
img1 = cv2.imread("img/shape1.jpg")
# fadsfds
rows, cols = img.shape[:2]
result = np.zeros((rows,cols,3), np.uint8)
CCW = np.array([[-1, -1], [-1, 0], [-1, 1],
                [0, 1], [1, 1],
                [1, 0], [1, -1], [0, -1]], np.int8)
# kernel = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], np.int8)
contour=[np.array([0,0])]
background = img[0,0,0]
middle = (np.max(img) + np.min(img))/2

def search_start(masiv):
    find = False

    print("Middle : ", middle)
    for i in range (rows):
        for j in range (cols):
            if abs(int(img[i,j,0]) - int(img[i,j-1,0])) >= middle:
                remember = img[i,j,0]
                # cv2.circle(result,(j,i), 1, (0,0,255), -1)
                result[i, j, :] = 255
                find = True
                boundaries(i,j, remember,masiv)
                break
        if find : break


def boundaries(i, j, shape_val, contour):
    startX, startY = i, j
    set_rotation = 0
    # contour=[[]]
    contour.append(np.array([startX, startY]))
    count = 1
    k=0
    check =False
    for l in range(len(CCW)):

            if img[i+CCW[l,0],j+CCW[l,1],0]<=middle :
                check =True
            if img[i+CCW[l,0],j+CCW[l,1],0] >=middle:
                while img[i+CCW[l,0],j+CCW[l,1],0] >=middle:
                    l+=1
                    if l==8:
                        l =0
                        break

                check =True

                if check ==True and img[i+CCW[l,0],j+CCW[l,1],0] <=middle:
                    contour.append(np.array([i,j]))
                    i=i+CCW[l,0]
                    j=j+CCW[l,1]
                    result[i, j, :] = 255
                    cv2.imshow("fdas", result)
                    check = False
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    l=0

            # l=0
            else:
                print("end")

    while i != startX or j !=startY :
       for l in range(len(CCW)):
            if img[i+CCW[l,0],j+CCW[l,1],0]<=middle :
                check =True
            if img[i+CCW[l,0],j+CCW[l,1],0] >=middle:
                while img[i+CCW[l,0],j+CCW[l,1],0] >=middle:
                    l+=1
                    if l==8:
                        l=0
                        break
                check =True
                if check ==True and img[i+CCW[l,0],j+CCW[l,1],0] <=middle:
                    contour.append(np.array([i,j]))
                    i=i+CCW[l,0]
                    j=j+CCW[l,1]
                    result[i, j, :] = 255
                    cv2.imshow("fdas", result)
                    cv2.waitKey(5)
                    check = False
                    l=0
                    if i == startX and j ==startY:
                        return
                    elif j == startX and i ==startY:
                        return
            if l==8:
                l=1
    cv2.imshow("fdas", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contour

search_start(contour)
contour = np.array(contour)
print(contour)
print(contour.shape)
# img1[3,3,0] = 0
# cv2.circle(img1,(4,4), 1, (0,0,255), -1)
# cv2.imshow("SLAVIK UKRAINE", result)
cv2.waitKey(0)
cv2.destroyAllWindows()