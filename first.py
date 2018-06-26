import cv2
import numpy as np

img = cv2.imread("1.jpg")


def enlargement(image):
    rows, cols = image.shape[:2]
    result = np.zeros((rows*2, cols*2, 3), np.uint8)
    for i in range(rows):
        for j in range(cols):
            kernel = image[i][j]
            result[i*2][j*2] = kernel
            result[i*2][(j*2)+1] = kernel
            result[(i*2)+1][j*2] = kernel
            result[(i*2)+1][(j*2)+1] = kernel
    cv2.imshow("2", result)


def reduction(image):
    rows, cols = image.shape[:2]
    result = np.zeros((rows//2, cols//2, 3), np.uint8)
    for i in range(rows//2):
        for j in range(cols//2):
            kernel = image[i*2][j*2]
            result[i][j] = kernel
    cv2.imshow("3", result)

enlargement(img)
reduction(img)
cv2.imshow("1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
