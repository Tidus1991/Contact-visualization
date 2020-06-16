# -*- coding: utf-8 -*-
"""
main.py

Created on 2020/6/14 10:31

@author: Tidus
"""

import cv2
import numpy as np


source = "contact.mp4"
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (540, 480))

class kPoint:
    def __init__(self, initArray):
        self.pt = (int(initArray[0]), int(initArray[1]))
        self.bestMatch = [None, np.inf]
        self.factor = None

    def matchPoint(self, ptArray, map_index):
        distanceFromMap = np.linalg.norm([self.pt[0]-ptArray[0], self.pt[1]-ptArray[1]])
        if distanceFromMap < self.bestMatch[-1]:
            self.bestMatch = [map_index, distanceFromMap]
            self.factor = np.exp(abs(distanceFromMap))/2

    def validityCheck(self,preFrame):
        minDistanceFromPreMap = np.inf
        for preKeyPoint in preFrame:
            tempDistance = np.linalg.norm([self.pt[0] - preKeyPoint[0], self.pt[1] - preKeyPoint[1]])
            if tempDistance < minDistanceFromPreMap:
                minDistanceFromPreMap  = tempDistance
        return minDistanceFromPreMap < 10 and self.bestMatch[-1] > 2.5 and self.bestMatch[-1] < 15

    def linearExpression(self, ptArray):
        theta = np.arctan2(ptArray[1]-self.pt[1], ptArray[0]-self.pt[0])
        endpt_x = int(ptArray[0] - self.factor * np.cos(theta))
        endpt_y = int(ptArray[1] - self.factor * np.sin(theta))
        return (endpt_x, endpt_y)


if __name__ == "__main__":
    frameIndex = 0
    kPointMap = np.empty((0, 2))
    preFrameMap = np.empty((0, 2))

    cap = cv2.VideoCapture(source)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    params = cv2.SimpleBlobDetector_Params()

    params.filterByConvexity = True
    params.minConvexity = 0.1

    params.filterByArea = True
    params.minArea = 30

    detector = cv2.SimpleBlobDetector_create(params)

    while cap.isOpened():
        _, rawFrame = cap.read()
        if rawFrame is None:
            break
        frameIndex+=1
        resizeFrame = cv2.resize(rawFrame,(640, 480))[:, 60:600]
        grayFrame = cv2.cvtColor(resizeFrame, cv2.COLOR_BGR2GRAY)

        binImg = cv2.adaptiveThreshold(grayFrame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15, 15)
        opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
        keyPonits = detector.detect(opening)
        im_with_keypoints = cv2.drawKeypoints(resizeFrame, keyPonits, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        if frameIndex == 30:
            for keyPoint in keyPonits:
                kPointMap = np.append(kPointMap, [[keyPoint.pt[0], keyPoint.pt[1]]], axis=0)
                preFrameMap = kPointMap
        if kPointMap.any():
            for keyPoint in keyPonits:
                currentPoint = kPoint(keyPoint.pt)
                for map_index in range(len(kPointMap)):
                    currentPoint.matchPoint(kPointMap[map_index], map_index)
                if currentPoint.validityCheck(preFrameMap):
                    linearDistance = currentPoint.linearExpression(kPointMap[currentPoint.bestMatch[0]])
                    cv2.arrowedLine(im_with_keypoints, tuple(kPointMap[currentPoint.bestMatch[0]].astype(np.int32)), linearDistance, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
                    print("Draw a arrowedLine")


            preFrameMap = np.empty((0, 2))
            for keyPoint in keyPonits:
                preFrameMap = np.append(preFrameMap, [keyPoint.pt], axis=0)
        out.write(im_with_keypoints)
        cv2.imshow("rawFrame", im_with_keypoints)
        print("frame:", frameIndex)

        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break