# Yoshika Takezawa
# 3/27/19
# Computer Vision hw1
# program relies on opencv, "pip install opencv-python"
# how to run: "python hw1.py <image>"

import math
import sys

import numpy as np
import cv2


def filter_funct(filter_matrix, msize, img):
    height, width = img.shape
    # create black image
    res = np.zeros((height, width), np.uint8)
    offby = msize//2
    for i in range(offby, height-offby):
        for j in range(offby, width-offby):
            fval = filterhelper(i, j, filter_matrix, msize, img)
            res[i][j] = fval
    return res


def filterhelper(x, y, fm, msize, img):
    total = 0.0
    offby = msize//2
    # iterating through each matrix element
    for i in range(msize):
        for j in range(msize):
            xval = x + i - offby
            yval = y + j - offby
            total += img[xval][yval] * fm[i][j]
    if (total <= 0):
        return 0
    elif (total >= 255):
        return 255
    return total


def nms(img):
    # applying vertical and horizontal sobel filters

    hsob = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
    I_x = filter_funct(hsob, 3, img)
    cv2.imwrite('x.png', I_x)

    vsob = [[1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]]
    I_y = filter_funct(vsob, 3, img)
    cv2.imwrite('y.png', I_y)

    height, width = img.shape

    gradient = np.power(np.power(I_x, 2.0) + np.power(I_y, 2.0), 0.5)
    theta = np.arctan2(I_x, I_y)

    cv2.imwrite('gradient.png', gradient)

    # Non-maximum suppression
    gradSup = gradient.copy()
    for i in range(height):
        for j in range(width):
            # image edge pixels become 0
            if (i == 0 or i == height-1 or j == 0 or j == width - 1):
                gradSup[i][j] = 0
                continue
            angle = theta[i][j] * 180.0/np.pi
            if (angle < 0):
                angle += 180
            if ((0 <= angle < 22.5) or (157.5 <= angle <= 180)):  # E-W (horizontal)
                if gradient[i][j] <= gradient[i][j-1] or gradient[i][j] <= gradient[i][j+1]:
                    gradSup[i][j] = 0
            elif (22.5 <= angle < 67.5):  # NE-SW
                if gradient[i][j] <= gradient[i-1][j+1] or gradient[i][j] <= gradient[i+1][j-1]:
                    gradSup[i][j] = 0
            elif (67.5 <= angle < 112.5):  # N-S (vertical)
                if gradient[i][j] <= gradient[i-1][j] or gradient[i][j] <= gradient[i+1][j]:
                    gradSup[i][j] = 0
            elif (112.5 <= angle < 157.5):  # NW-SE
                if gradient[i][j] <= gradient[i-1][j-1] or gradient[i][j] <= gradient[i+1][j+1]:
                    gradSup[i][j] = 0
    cv2.imwrite('out.png', gradSup)
    return gradSup


def threshold(img, high=0.06, low=0.02):

    highThreshold = img.max() * high
    lowThreshold = highThreshold * low

    height, width = img.shape
    res = np.zeros((height, width), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    cv2.imwrite('thresh.png', res)
    return (res, weak, strong)


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        sys.exit("usage: python hw1.py <img>")
    orig = cv2.imread(sys.argv[1], 0)

    print("Working on it!")

    # applying gaussian filter
    gaussian = [[0.003, 0.013, 0.022, 0.013, 0.003],
                [0.013, 0.059, 0.097, 0.059, 0.013],
                [0.022, 0.097, 0.159, 0.097, 0.022],
                [0.013, 0.059, 0.097, 0.059, 0.013],
                [0.003, 0.013, 0.022, 0.013, 0.003]]
    g_res = filter_funct(gaussian, 5, orig)
    cv2.imwrite('gauss.png', g_res)

    nms_res = nms(g_res)
    threshold(nms_res)

    print("Done!")