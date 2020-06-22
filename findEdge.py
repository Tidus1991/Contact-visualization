# -*- coding: utf-8 -*-
"""
Created on 2020/6/18 13:44

@author: Tidus
"""

import cv2
import numpy as np

class FindEdge:
    def __init__(self, thresholds=254):
        self.thr = thresholds

    def cntArea(self, cnt):
        area = cv2.contourArea(cnt)
        return area

    def findEdge(self, image):
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,2]
        _, frame = cv2.threshold(frame, self.thr, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # binImg = cv2.adaptiveThreshold(frame,255,cv2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11, 8)
        contours.sort(key = self.cntArea, reverse=False)
        for cnt in contours:
            if self.cntArea(cnt)>2000:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cx >50 and cy>50 and cx<500:
                    results = cv2.drawContours(image, cnt, -1, (0, 0, 255), 2)
                    break
            else:
                results = image
        return results

if __name__ == "__main__":
    edgeFinder = FindEdge()
    cap = cv2.VideoCapture("vidData/contact_v4_2.mp4")

    while cap.isOpened():
        _, rawImg = cap.read()
        if rawImg is None:
            break
        processedImg = edgeFinder.findEdge(cv2.resize(rawImg,(640,480)))
        cv2.imshow("result", processedImg)
        if cv2.waitKey(30) & 0xff == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
