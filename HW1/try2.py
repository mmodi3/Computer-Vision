import PIL
import cv2
import numpy as np
from cv2 import *

""" Kernel for filter = 
    .075    .124    .075
    .124    .203    .124
    .075    .124    .075 
    
    Sobel Vertical = 
    1 0 -1 
    2 0 -2 
    1 0 -1
    
    Sobel Horizontal = 
     1  2  1
     0  0  0 
    -1 -2 -1"""

def xSuppresion(res, point):
    height, width = res.shape
    pixelVal = res[point[0]][point[1]]
    if point[0] < 1 or point[0] > height - 2:
        return res
    for offset in range(-1, 2):
        if offset == 0:
            continue
        if res[point[0]+offset][point[1]] > res[point[0]][point[1]]:
            res[point[0]][point[1]] = 0
            return res
    for offset in range(-1, 2):
        res[point[0]+offset][point[1]] = 0
    res[point[0]][point[1]] = pixelVal
    return res

def ySuppresion(res, point):
    height,width = res.shape
    pixelVal = res[point[0]][point[1]]
    if point[1] < 1 or point[1] > width - 2:
        return res
    for offset in range(-1, 2):
        if offset == 0:
            continue
        if res[point[0]][point[1]+offset] > res[point[0]][point[1]]:
            res[point[0]][point[1]] = 0
            return res
    for offset in range(-1, 2):
        res[point[0]][point[1]+offset] = 0
    res[point[0]][point[1]] = pixelVal
    return res

def hessianNMS(img, point):
    offset = 2
    height, width = img.shape
    if point[1] <= offset or point[1] >= width - offset - 1 or point[0] <= offset or point[0] >= height - 1 - offset:
        return img
    for i in range(-offset, offset+1):
        for j in range(-offset, offset+1):
            if img[point[0]][point[1]] < img[point[0]+i][point[1]+j]:
                img[point[0]][point[1]] = 0
                return img
            else:
                img[point[0]+i][point[1]+j] = 0
    img[point[0]][point[1]] = 255
    return img

def gaussian(img):
    corners = 0.077847
    sides = 0.123317
    center = 0.195346
    height, width = img.shape
    res = img.copy()
    for j in range(1, width-1):
        for i in range(1, height-1):
            pixelVal = img[i-1][j-1]*corners + img[i-1][j+1]*corners + img[i+1][j+1]*corners + img[i+1][j-1]*corners \
                + img[i][j-1]*sides + img[i][j+1]*sides + img[i-1][j]*sides + img[i+1][j]*sides + img[i][j]*center
            res[i][j] = int(pixelVal)
    return res

def Gx(img):
    height,width = img.shape
    res = img.copy()
    points = []
    for j in range(1, width-1):
        for i in range(1, height-1):
            pixelVal = img[i-1][j-1] + 2*img[i-1][j] + img[i-1][j+1] - (img[i+1][j-1] + 2*img[i+1][j] + 1*img[i+1][j-1])
            # pixelVal = 5*(img[i-2][j-2] + img[i-2][j+2] - img[i+2][j-2] - img[i+2][j+2]) + 4*(img[i-2][j-1] + img[i-2][j+1] - img[i+2][j-1] - img[i+2][j+1]) + \
            #     8*(img[i-1][j-2] + img[i-1][j+2] - img[i+1][j-2] - img[i+1][j+2]) + 10*(img[i-1][j-1] + img[i-1][j+1] - img[i+1][j-1] - img[i+1][j+1]) + \
            #         10*(img[i-2][j] - img[i+2][j]) + 20*(img[i-1][j] - img[i+1][j])
            if pixelVal > img[i][j]:
                res[i][j] = pixelVal
                points += [[i,j]]
            else:
                res[i][j] = 0
    for point in points:
        res = xSuppresion(res, point)
    return res

def Gy(img):
    height,width = img.shape
    res = img.copy()
    points = []
    for j in range(1, width-1):
        for i in range(1, height-1):
            pixelVal = img[i-1][j-1] + 2*img[i][j-1] + img[i+1][j-1] - (img[i-1][j+1] + 2*img[i][j+1] + img[i+1][j+1])
            # pixelVal = img[i-2][j-2] + img[i-2][j+2] + img[i+2][j-2] + img[i+2][j+2] + 2*(img[i-2][j-1] + img[i-2][j+1] + img[i+2][j-1] + img[i+2][j+1]) + \
            #     4*(img[i-1][j-2] + img[i-1][j+2] + img[i+1][j-2] + img[i+1][j+2]) + 8*(img[i-1][j-1] + img[i-1][j+1] + img[i+1][j-1] + img[i+1][j+1]) + \
            #         6*(img[i][j-2] + img[i][j+2]) + 12*(img[i][j-1] + img[i][j+1])
            if pixelVal > img[i][j]:
                res[i][j] = pixelVal
                points += [[i,j]]
            else:
                res[i][j] = 0
    for point in points:
        res = ySuppresion(res, point)
    return res
    
def composite(img1, img2):
    height, width = img1.shape
    res = img1.copy()
    for i in range(1, height-1):
        for j in range(1, width-1):
            if res[i][j] > 0:
                res[i][j] = 75
            if img2[i][j] > 0:
                res[i][j] = 75
    return res

def gradient(img):
    return [Ix, Iy]

def hessian(img):
    height,width = img.shape
    k = .04
    thresh = 10
    Ix, Iy = gradient(img)
    Ixy = Ix * Iy
    Ixx = Ix**2
    Iyy = Iy**2
    comp = composite(Ix, Iy)
    res = comp.copy()*0
    offset = 2
    points = []
    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            Sxx = np.sum(Ixx[i-offset:i+1+offset, j-offset:j+1+offset])
            Syy = np.sum(Iyy[i-offset:i+1+offset, j-offset:j+1+offset])
            Sxy = np.sum(Ixy[i-offset:i+1+offset, j-offset:j+1+offset])
            det =Sxx*Syy - Sxy**2
            trace = Sxx+Syy
            r = det - k*(trace**2)
            if r > 0:
                res[i][j] = int(r)
                points += [(i, j)]
                for x in range(-5, 6):
                    for y in range(-5, 6):
                        if i+x > height-1 or i+x < 0 or j+y > width-1 or j+y < 0:
                            continue
                        res[i+x][j+y] = 0
    for point in points:
        if res[point[0]][point[1]] == 0:
            points.remove(point)
        else:
            res = hessianNMS(res, point)
            if res[point[0]][point[1]] == 0:
                points.remove(point)
    for i in range(1, height-1):
        for j in range(1, width-1):
            if res[i][j] != 255:
                res[i][j] = comp[i][j]
    return res

def distance(p1, p2, p3):
    if p1 == p2:
        return np.sqrt((p2[0]-p3[0])**2 + (p2[1] - p3[1])**2)
    p1=np.array(p1)
    p2=np.array(p2)
    p3=np.array(p3)
    dist = np.linalg.norm(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)
    return dist

def lineseperation(start, end, points, thresh):
    for point in points:
        if distance(start, end, point) < thresh:
            return True
    return False

def slope(p1, p2):
    return float((p2[1] - p1[1])/(p2[0] - p1[0]))

def fullline(img, p1, p2, color):
    height, width, idk = img.shape
    if p1[0] == p2[0]:
        img = cv2.line(img, tuple(0, p1[1]), tuple(width, p2[1], color, 1))
    slp = slope(p1, p2)
    if p1[0] < p2[0]:
        while p1[0] >= 0:
            p1[0] -= 1
            p1[1] -= slp
        while p2[0] <= width:
            p2[0] += 1
            p2[1] += slp
    if p1[0] > p2[0]:
        while p1[0] <= width:
            p1[0] += 1
            p1[1] += slp
        while p2[0] >= 0:
            p2[0] -= 1
            p2[1] -= slp
    p1[0] = int(p1[0])
    p2[0] = int(p2[0])
    p1[1] = int(p1[1])
    p2[1] = int(p2[1])
    img = cv2.line(img, tuple(p1), tuple(p2), color, 1)
    return img

def ransac(img):
    height, width = img.shape
    res = img.copy()
    points = []
    for i in range(1, height-1):
        for j in range(1, width-1):
            if res[i][j] == 255:
                points += [[i,j]]
    res = cv2.cvtColor(res, COLOR_GRAY2BGR)
    numlines = 0
    proximity = 10
    pointsneeded = 10
    colors = [[0, 0, 255], [255, 255, 0], [0, 255, 0], [0, 255, 255]]
    linesused = []
    while numlines < 4:
        pointsused = []
        start = points[np.random.randint(0, len(points)-1)]
        end = points[np.random.randint(0, len(points)-1)]
        if linesused:
            while lineseperation(start, end, linesused, 15):
                start = points[np.random.randint(0, len(points)-1)]
                end = points[np.random.randint(0, len(points)-1)]
        pointsclose = 2
        for point in points:
            dist = distance(start, end, point)
            if dist < proximity:
                pointsused += [point]
                pointsclose += 1
        if pointsclose >= pointsneeded:
            for point in pointsused:
                points.remove(point)
                res[point[0]][point[1]] = res[point[0]-1][point[1]-1] = res[point[0]-1][point[1]] = res[point[0]-1][point[1]+1] = res[point[0]][point[1]-1] = res[point[0]][point[1]+1]= res[point[0]+1][point[1]-1] = res[point[0]+1][point[1]] = res[point[0]+1][point[1]+1] = colors[numlines]
            maxdist = 0
            maxpair = []
            for i in range(pointsclose):
                for j in range(i, pointsclose):
                    dist = distance(pointsclose[i], pointsclose[i], pointsclose[j])
                    if dist > maxdist:
                        maxdist = dist
                        maxpair = [pointsclose[i], pointsclose[j]]
            # res = fullline(res, [start[1], start[0]], [end[1], end[0]], tuple(colors[numlines]))
            res = cv2.line(res, tuple(maxpair[0]), tuple(maxpair[1]), colors[numlines], 1)
            numlines += 1
            linesused += [start, end]
    for point in linesused:
        res[point[0]][point[1]] = [255, 0, 0]
        for i in range(-2,3):
            for j in range(-2,3):
                res[point[0]+i][point[1]+j] = [255,0,0]
    return res

if __name__ == '__main__':
    img = imread('road.png', 0)
    print("Filtering Started...")
    filtered = gaussian(img)
    print("Detecting Horizontal Lines...")
    horizontal = Gx(filtered)
    imwrite('horizontal.png', horizontal, [IMWRITE_PNG_COMPRESSION, 0])
    print("Detecting Vertical Lines...")
    vertical = Gy(filtered)
    imwrite('vertical.png', vertical, [IMWRITE_PNG_COMPRESSION, 0])
    print("Creating Composite Image...")
    comp = composite(horizontal, vertical)
    imwrite('composite.png', comp, [IMWRITE_PNG_COMPRESSION, 0])
    print("Detecting Corners...")
    hess = hessian(filtered)
    imwrite('corners.png', hess[1], [IMWRITE_PNG_COMPRESSION, 0])
    print("Running RANSAC...")
    rnsc = ransac(hess)
    imwrite('ransac.png', rnsc, [IMWRITE_PNG_COMPRESSION, 0])
    
    


