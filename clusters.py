# -*- coding: utf-8 -*-
"""
Created on 2020/06/16 13:29

@author: Tidus
"""

import cv2
import numpy as np

colorList = np.uint8([[255, 0, 0],
                          [0, 0, 255],
                          [128, 128, 128],
                          [0, 255, 0],
                         [128, 128, 0],
                         [128, 0, 128],
                         [0, 128, 128]])

class clusters:
    def __init__(self, numClusters):
        self.num_clusters = numClusters

    def Clustering(self, image):
        maxColor = -np.inf
        contactClass = None

        rawImg = image
        reshapeImg = np.float32(rawImg.reshape((-1, 3)))
        h, w, ch = rawImg.shape
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(reshapeImg, self.num_clusters, None, criteria, self.num_clusters, cv2.KMEANS_RANDOM_CENTERS)

        for i, j in enumerate(center):
            if  j[0] > 60 and j[1] > 220 and  maxColor <= j[1]:
                maxColor = j[1]
                contactClass = i
        label = np.reshape(label, (h, w))
        if contactClass is not None:
        #     return [label == contactClass]
        # else:
        #     return False
            rawImg[label == contactClass] = colorList[1]
        return rawImg


if __name__ == "__main__":
    twoClusters = clusters(5)
    cap = cv2.VideoCapture("vidData/contact_v4_2.mp4")

    while cap.isOpened():
        _, rawImg = cap.read()
        if rawImg is None:
            break
        processedImg = twoClusters.Clustering(cv2.resize(rawImg,(640,480)))
        cv2.imshow("result", processedImg)
        if cv2.waitKey(30) & 0xff == 27:
            break
    cap.release()
    cv2.destroyAllWindows()