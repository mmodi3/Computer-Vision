# Mitra Modi
# 3/27/19
# Computer Vision HW1
# Program has dependencies on the numpy and opencv libraries, which must be installed to run the program
# I pledge my honor that I have abided by the stevens honor system - Mitra Modi

import cv2
import numpy as np
from cv2 import *

#Function used to apply a kernel defined as a 2-D array on an image
def filteringFunction(filtermatrix, msize, img):
    height, width = img.shape
    # create blank image
    res = np.zeros((height, width), np.uint8)
    offset = msize//2
    #Iterate through region of image where window falls completely within the image
    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            fval = filterhelper(i, j, filtermatrix, msize, img)
            res[i][j] = fval
    return res



#Helper function for filteringFunction, applies the kernel to the image at coordinate defined by x and y
def filterhelper(x, y, filtermatrix, msize, img):
    total = 0.0
    offset = msize//2
    # iterating through each element in the neighborhood of the pixel at (x,y) corresponding to the kernel
    for i in range(msize):
        for j in range(msize):
            xval = x + i - offset
            yval = y + j - offset
            total += img[xval][yval] * filtermatrix[i][j]
    #Threshold lower bound
    if (total <= 0):
        return 0
    #Threshold upper bound
    elif (total >= 255):
        return 255
    return total



#Helper function for the sobel filters, used to remove diagonal lines and reduce the thickness of detected lines to make the lines easier to use
def NonMaxSup(Ix, Iy):
    height,width = img.shape
    #find gradient to be used in suppressing the x and y derivatives
    gradient = np.power(np.power(Ix, 2.0) + np.power(Iy, 2.0), 0.5)
    theta = np.arctan2(Ix, Iy)
    for i in range(height):
        for j in range(width):
            # image edge pixels become 0
            if (i == 0 or i == height-1 or j == 0 or j == width - 1):
                Ix[i][j] = Iy[i][j] = 0
                continue
            angle = theta[i][j] * 180.0/np.pi
            if (angle < 0):
                angle += 180
            #Considered to be a horizontal line
            if ((0 <= angle < 22.5) or (157.5 <= angle <= 180)):
                if gradient[i][j] <= gradient[i][j-1] or gradient[i][j] <= gradient[i][j+1]:
                    Ix[i][j] = Iy[i][j] = 0
            #Condiered to be a diagonal line
            elif (22.5 <= angle < 67.5): 
                if gradient[i][j] <= gradient[i-1][j+1] or gradient[i][j] <= gradient[i+1][j-1]:
                    Ix[i][j] = Iy[i][j] = 0
            #Considered to be a vertical line
            elif (67.5 <= angle < 112.5):
                if gradient[i][j] <= gradient[i-1][j] or gradient[i][j] <= gradient[i+1][j]:
                    Ix[i][j] = Iy[i][j] = 0
            #Considered to be a diagonal line
            elif (112.5 <= angle < 157.5):
                if gradient[i][j] <= gradient[i-1][j-1] or gradient[i][j] <= gradient[i+1][j+1]:
                    Ix[i][j] = Iy[i][j] = 0
    return [Ix, Iy]



#Helper function for sobel filter used to threshold the values, making the lines easier to use
def threshold(img, high=0.20, low=0.02):
    upper = img.max() * high #Calculate upper threshold for values considered via the maximum value in the image
    lower = upper * low #Calculate lower threshold for values consdered via minimuum
    height, width = img.shape
    res = np.zeros((height, width), dtype=np.int32) #Initiate blank image of same size as img
    weak = 0 #Pixel val for weak pixels
    strong = 255 #Pixel val for strong pixels
    strong_i, strong_j = np.where(img >= upper)
    zeros_i, zeros_j = np.where(img < lower)
    weak_i, weak_j = np.where((img <= upper) & (img >= lower))
    res[strong_i, strong_j] = img[strong_i, strong_j]
    res[weak_i, weak_j] = weak
    return res



#Function used to create a composite image of the sobelx and sobely, used to place as a background for ransac and hough
def composite(Ix, Iy):
    res = Ix.copy()
    height,width = Ix.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            res[i][j] = 50 if res[i][j] > 0 or Iy[i][j] > 0 else 0
    return res



#Helper function for hessian to suppress the feature points to limit repeated detections
def hessianNMS(img, point):
    if img[point[0]][point[1]] == 0:
        return img
    offset = 1
    height, width = img.shape
    if point[1] <= offset or point[1] >= width - offset - 1 or point[0] <= offset or point[0] >= height - 1 - offset: #Make sure point is within region where filter lies fully in the region
        return img
    for i in range(-offset, offset+1):
        for j in range(-offset, offset+1):
            if img[point[0]][point[1]] < img[point[0]+i][point[1]+j]: #Non Maximum Suppression
                img[point[0]][point[1]] = 0
                return img
            else:
                img[point[0]+i][point[1]+j] = 0
    img[point[0]][point[1]] = 255
    return img



#Function to find feature points from sobelx and sobely gradients
def hessian(img, Ix, Iy):
    sobelxKernel = [[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]]

    sobelyKernel = [[ 1,  0, -1],
                    [ 2,  0, -2],
                    [ 1,  0, -1]]
    height,width = img.shape
    k = .06
    thresh = 300000

    #For second derivatives, I found 300000 to be a good valuem for derivatives squared, 1250000000

    # Ixy = Ix * Iy
    # Ixx = Ix**2
    # Iyy = Iy**2

    #COMMENT OUT UPPER BLOCK AND USE LOWER FOR SECOND DERIVATIVE BASED HESSIAN, USE TOP FOR DERIVATIVE SQUARED BASED

    Ixy = filteringFunction(sobelxKernel, 3, Iy)
    Iyy = filteringFunction(sobelyKernel, 3, Iy)
    Ixx = filteringFunction(sobelxKernel, 3, Ix)

    res = img.copy()*0
    offset = 2
    points = []
    #Iterate through portion of the image where the window is completely inside the image
    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            Sxx = np.sum(Ixx[i-offset:i+1+offset, j-offset:j+1+offset]) #Window sum of Ixx
            Syy = np.sum(Iyy[i-offset:i+1+offset, j-offset:j+1+offset]) #Window sum of Iyy
            Sxy = np.sum(Ixy[i-offset:i+1+offset, j-offset:j+1+offset]) #Window sum of Ixy
            det =Sxx*Syy - Sxy**2
            trace = Sxx+Syy
            r = det - k*(trace**2)
            if r > thresh:
                res[i][j] = r
    #Non Maximum Suppression for each pixel of the calculated hessian
    for i in range(1, height-1):
        for j in range(1, width-1):
            res = hessianNMS(res, [i,j])
    #Threshold values to give bright corners
    for i in range(1, height-1):
        for j in range(1, width-1):
            res[i][j] = res[i][j] if res[i][j] > img[i][j] else img[i][j]
    return res



#Helper function for RANSAC, used to determine distance from a line to a point
def distance(p1, p2, p3):
    if p1 == p2:
        return np.sqrt((p2[0]-p3[0])**2 + (p2[1] - p3[1])**2)
    p1=np.array(p1)
    p2=np.array(p2)
    p3=np.array(p3)
    dist = np.linalg.norm(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)
    return dist



#Helper function for RANSAC, helps ensure that two lines are not chosen to be too close to one another
def lineseperation(start, end, points, thresh):
    for point in points:
        if distance(start, end, point) < thresh:
            return True
    return False



#Function that applies the RANSAC algorithm on the feature points found by the hessian
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
        pointsused += [start, end]
        pointsclose = 2
        for point in points:
            dist = distance(start, end, point)
            if dist < proximity:
                pointsused += [point]
                pointsclose += 1
        if pointsclose >= pointsneeded:
            for point in pointsused:
                res[point[0]][point[1]] = res[point[0]-1][point[1]-1] = res[point[0]-1][point[1]] = res[point[0]-1][point[1]+1] = res[point[0]][point[1]-1] = res[point[0]][point[1]+1]= res[point[0]+1][point[1]-1] = res[point[0]+1][point[1]] = res[point[0]+1][point[1]+1] = colors[numlines]
                if point == start or point == end:
                    continue
                points.remove(point)
            maxdist = distance(start, start, end)
            maxpair = [start, end]
            for i in range(len(pointsused)):
                for j in range(len(pointsused)):
                    dist = distance(pointsused[i], pointsused[i], pointsused[j])
                    if dist > maxdist:
                        maxdist = dist
                        maxpair = [pointsused[i], pointsused[j]]
            res = cv2.line(res, tuple([maxpair[0][1], maxpair[0][0]]), tuple([maxpair[1][1], maxpair[1][0]]), colors[numlines], 1)
            numlines += 1
            linesused += maxpair
    for point in linesused:
        res[point[0]][point[1]] = [255, 0, 0]
        for i in range(-2,3):
            for j in range(-2,3):
                res[point[0]+i][point[1]+j] = [255,0,0]
    return res



#Helper function for Hough Transform that detects the bins with the most points, allowing for the drawing of the best supported lines
def houghbestlines(lst):
    rows = len(lst)
    cols = len(lst[0])
    first = second = third = fourth = [0]
    for i in range(rows):
        for j in range(cols):
            curr = [lst[i][j], i, j]
            if curr[0] > first[0]:
                temp1 = first
                first = curr
                temp2 = second
                second = temp1
                temp1 = third
                third = temp2
                temp2 = fourth
                fourth = temp1
            elif curr[0] > second[0]:
                temp2 = second
                second = curr
                temp1 = third
                third = temp2
                fourth = temp1
            elif curr[0] > third[0]:
                temp1 = third
                third = curr
                fourth = temp1
            elif curr[0] > fourth[0]:
                fourth = curr
    return [first, second, third, fourth]



#Helper function for full line, used to calculate floating point value of the slope between two points
def slope(p1, p2):
    return float((p2[1] - p1[1])/(p2[0] - p1[0]))



#Helper function for drawhoughline, used to draw full line instead of a line segment on the image, utilizes inbuilt function cv2.line()
def fullline(img, p1, p2, color):
    height, width, idk = img.shape
    if p1[0] == p2[0]:
        img = cv2.line(img, tuple(0, p1[1]), tuple(width, p2[1], color, 1)) #Edge case where x vals are the same, using slope function would through zero division error
    slp = slope(p1, p2)

    #Loops to increment/decrement point values to get beyond the boundries of the image
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
    img = cv2.line(img, tuple(p1), tuple(p2), color, 1) #Draws line with points outside the boundries to make spanning line
    return img

#Helper function for Hough Transform that draws a line defined by two parameters, rho and theta
def drawhoughline(img, line):
    r = line[1] #RHO value
    theta = np.deg2rad(line[2]) #THETA value
    b = np.cos(theta)
    a = np.sin(theta)
    x0 = a*r 
    y0 = b*r 
    x1 = int(x0 + 1000*(-b)) #X value of "initial point" used in graphing the line
    y1 = int(y0 + 1000*(a))  #Y value of "initial point" used in graphing the line
    x2 = int(x0 - 1000*(-b)) #X value of "final point" used in graphing the line
    y2 = int(y0 - 1000*(a))  #Y value of "final point" used in graphing the line
    img = fullline(img, [x1,y1], [x2,y2], (0,0,255)) #Helper function to draw the full line from a line segment
    return img

# Function used on the set of detected feature points to find the most supported lines in the image
def hough_line(img): 
    height, width = img.shape #Gets height and width of the img
    res = img.copy()
    res = cv2.cvtColor(res, COLOR_GRAY2BGR) #Creates a 3 band (RGB) copy of the passed image, does not modify how it looks, just how it is processed (Pixel vals look like [B,R,G] vs 0-255)
    diagonal = int((width**2 + height**2)**.5) #Calculates diaognal length of image
    thetas = np.arange(0, 180, 2) #Used to create the range of thetas, as well as the gap between them in order to tweak the parameters for hough
    accumulator = np.zeros((int(2 * diagonal), 181)) #Creates the empty accumulator to store votes in bins
    points = []
    for i in range(1, height-1): #Loop to determine where the feature points are, making the process easier
        for j in range(1, width-1):
            if img[i][j] == 255:
                points += [[i,j]] 
    for point in points:
        x = point[0]
        y = point[1]
        for theta in thetas:
            rho = x*np.cos(np.deg2rad(theta)) + y*np.sin(np.deg2rad(theta)) #Calculates rho
            accumulator[int(rho)][theta] += 1 #Increments bind to act as a vote
    bestlines = houghbestlines(accumulator) #Finds four most supported lines
    for line in bestlines:
        res = drawhoughline(res, line) #Draws infinite lines from the most supported lines provided
    return res


if __name__ == '__main__':
    colors = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]]
    gaussianKernel = [[0.077847, 0.123317, 0.077847],
                      [0.123317, 0.195346, 0.123317],
                      [0.077847, 0.123317, 0.077847]]
    
    sobelxKernel = [[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]]

    sobelyKernel = [[ 1,  0, -1],
                    [ 2,  0, -2],
                    [ 1,  0, -1]]

    userInput = input("Specify the file you would like to process: ")
    img = imread(userInput, 0)
    height, width = img.shape
    print('Filtering the image using Gaussian kernel...')
    gaussian = filteringFunction(gaussianKernel, 3, img)
    cv2.imwrite('gauss.png', gaussian, [IMWRITE_PNG_COMPRESSION, 0])
    print('Detecting horizontal lines...')
    x = filteringFunction(sobelxKernel, 3, gaussian)
    print('Detecting vertical lines...')
    y = filteringFunction(sobelyKernel, 3, gaussian)
    print('Completing Non Maximum Supression...')
    x,y = NonMaxSup(x,y)
    print("Thresholding...")
    x = threshold(x, .3)
    y = threshold(y, .3)
    comp = composite(x,y)
    cv2.imwrite('x.png', x, [IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('y.png', y, [IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('composite.png', comp, [IMWRITE_PNG_COMPRESSION, 0])
    print('Detecting Corners...')
    hess = hessian(comp, x, y)
    cv2.imwrite('corners.png', hess, [IMWRITE_PNG_COMPRESSION, 0])
    print('Running RANSAC...')
    hess = imread('corners.png', 0)
    rnsc = ransac(hess)
    cv2.imwrite('ransac.png', rnsc, [IMWRITE_PNG_COMPRESSION, 0])
    print('Running Hough Transform...')
    hough = hough_line(hess)
    cv2.imwrite('houghnobackground.png', hough, [IMWRITE_PNG_COMPRESSION, 0])
    img = cv2.cvtColor(img, COLOR_GRAY2BGR)
    cv2.imwrite('houghbackground.png', hough, [IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('ransacbackground.png', rnsc, [IMWRITE_PNG_COMPRESSION, 0])