import cv2
import numpy as np

def houghTransform(edgeMap, p_max):

    (height, width) = edgeMap.shape[:2]
    degrees = 180
    houghImage = np.zeros((2*(p_max+1)+1, degrees+1), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            if edgeMap[y][x] != 0:
                for theta in range(1,degrees):
                    p = x*np.cos(theta*np.pi/180) + y*np.sin(theta*np.pi/180)
                    p = int(p + p_max)
                    if(houghImage[p][theta] < 255):
                        houghImage[p][theta] += 1

    return houghImage


def findPointsForLine(p, theta, width, height):
    x_0 = 0
    y_0 = 0
    try:
        y_0 = int(p/np.sin(theta*np.pi/180))
    except ZeroDivisionError:
        x_0 = p
        y_0 = 0

    x_1 = width - 1
    y_1 = 0
    try:
        y_1 = int((p - x_1 * np.cos(theta * np.pi / 180)) / np.sin(theta * np.pi / 180))
    except ZeroDivisionError:
        x_1 = p
        y_1 = 1

    y_2 = 0
    x_2 = 0
    try:
        x_2 = int(p/np.cos(theta*np.pi/180))
    except ZeroDivisionError:
        y_2 = p
        x_2 = 0

    y_3 = height - 1
    x_3 = 0
    try:
        x_3 = int((p - y_3 * np.sin(theta*np.pi/180))/np.cos(theta * np.pi/180))
    except ZeroDivisionError:
        y_3 = p
        x_3 = 1

    points = np.zeros((4,2), dtype=np.uint8)
    k = 0
    if y_0 >= 0 and y_0 < height:
        points[k][0] = np.uint8(x_0)
        points[k][1] = np.uint8(y_0)
        k += 1

    if y_1 >= 0 and y_1 < height:
        points[k][0] = np.uint8(x_1)
        points[k][1] = np.uint8(y_1)
        k += 1

    if x_2 >= 0 and x_2 < width:
        points[k][0] = np.uint8(x_2)
        points[k][1] = np.uint8(y_2)
        k += 1

    if x_3 >= 0 and x_3 < width:
        points[k][0] = np.uint8(x_3)
        points[k][1] = np.uint8(y_3)
        k += 1
    return points


if __name__ == "__main__":
    img = cv2.imread("./hough/hough1.png")
    cannyEdgeImg = cv2.Canny(img,100,200)
    (height, width) = cannyEdgeImg.shape[:2]
    p_max = int(np.sqrt(height**2 + width**2))
    houghImage = houghTransform(cannyEdgeImg, p_max)
    threshold = 40

    lineDict = {}
    (houghHeight, houghWidth) = houghImage.shape[:2]
    for p in range(houghHeight):
        for t in range(houghWidth):
            if houghImage[p][t] >= threshold:
                if p not in lineDict:
                    lineDict[p-p_max] = []
                lineDict[p-p_max].append(t)

    print(lineDict)
    i = 0
    for p in lineDict:
        for theta in lineDict[p]:
            points = findPointsForLine(p, theta, width, height)
            cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (255,255), 1)
    cv2.imshow("Edge Map", cannyEdgeImg)
    cv2.imshow("Hough Map", houghImage)
    cv2.imshow("Imagem das Linhas", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()