import cv2
import numpy as np

img = cv2.imread("1.jpg")


def filter(image, size, sigma):
    # mask = [3][3]
    mask = np.zeros((size, size), np.float64)
    # fill_kernel(size, sigma, mask)
    mask = cv2.getGaussianKernel(size*size, sigma)
    print(mask)
    # mask = [1, 2, 1,
    #         2, 4, 2,
    #         1, 2, 1]
    rows, cols = image.shape[:2]
    suma = 0
    for i in range(size//2, rows-size//2):
            for j in range(size//2, cols-size//2):
                    for a in range(0, size):
                            for b in range(0, size):
                                suma += image[(i+a-size/2) + (j+b-size/2)] * mask[a*size+b]




def value(x, y, s):
    # print(1/(2*np.pi*pow(s, 2)) * np.exp(-(pow(x, 2) + pow(y, 2))/(2*pow(s, 2))))
    return 1/(2*np.pi*pow(s, 2)) * np.exp(-(pow(x, 2) + pow(y, 2))/(2*pow(s, 2)))


def fill_kernel(size, s, mask):
    suma = 0
    suma = float(suma)
    for i in range((size//2), (-size//2), -1):
        print("i:", i)
        for j in range((size//2), (-size//2), -1):
            print("j:", j)
            mask[i][j] = value(i, j, s)
            # print(mask[i][j])
            # suma += mask[i][j]
    print("suma1:", suma, "mask1:\n", mask)
    # for i in range((-size//2)-1, size//2):
    #     for j in range((-size//2)-1, size//2):
    #         mask[i][j] /= suma
    # print("suma2:", suma, "mask2:\n", mask)
    return mask

# filter(img, 5, 0.73876)
filter(img, 3, 1)