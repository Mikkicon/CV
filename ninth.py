import cv2
# import time
import numpy as np
# from matplotlib import pyplot as plt

img = cv2.imread("img/shape.jpg")
img1 = cv2.imread("img/shape1.jpg")
rows, cols = img.shape[:2]
result = np.zeros((rows,cols,3), np.uint8)
# cv2.imshow("SLAVIK UKRAINE", img1)
kernel = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], np.int8)
print(kernel.shape)


def search_start():
    find = False
    middle = (np.max(img) + np.min(img))/2
    print("Middle : ", middle)
    for i in range (rows):
        for j in range (cols):
            if abs(int(img[i,j,0]) - int(img[i,j-1,0])) >= middle:
                remember = img[i,j,0]
                # cv2.circle(result,(j,i), 1, (0,0,255), -1)
                result[i, j, :] = 255
                # cv2.imshow("SLAVIK UKRAINE", result)
                find = True
                boundaries(i,j, remember)
                break
        if find : break


def boundaries(i, j, shape_val):
    startX, startY = i, j
    set_rotation = 0
    # i, j = startX, startY
    count = 0
    while i != startX & j !=startY:
        # for a in range  (kernel.shape[0]):
            if(set_rotation == 0):
                while img[i-1][j-1][0] != shape_val | count == 2:
                    count +=1
                    if img[i-1][j-1][0] == shape_val:
                        result[i-1][j-1][0] = 255
                        i, j = i-1, j-1
                set_rotation = 6
                count=0
            elif(set_rotation == 6):
                while img[i-1][j+1][0] != shape_val | count == 2:
                    count +=1
                    if img[i-1][j+1][0] == shape_val:
                        result[i-1][j+1][0] = 255
                        i, j = i-1, j+1
                set_rotation = 8
                count=0
            elif(set_rotation == 8):
                while img[i+1][j+1][0] != shape_val | count == 2:
                    count +=1
                    if img[i+1][j+1][0] == shape_val:
                        result[i+1][j+1][0] = 255
                        i, j = i+1, j+1
                set_rotation = 2
                count=0
            elif(set_rotation == 2):
                while img[i+1][j-1][0] != shape_val | count == 2:
                    count +=1
                    if img[i+1][j-1][0] == shape_val:
                        result[i+1][j-1][0] = 255
                        i, j = i+1, j-1
                set_rotation = 0
                count=0


# cv2.imwrite("img/shape1.jpg", img1)

search_start()
# img1[3,3,0] = 0
# cv2.circle(img1,(4,4), 1, (0,0,255), -1)
cv2.imshow("SLAVIK UKRAINE", result)
cv2.waitKey(0)
cv2.destroyAllWindows()