#!/usr/bin/env python -W ignore::DeprecationWarning

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import uuid


class Tracker:
    def __init__(self, frame, roi, track_window, timeout_min=0.5):
        self.timeout = time.time() + 60*timeout_min

        self.hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv_roi, np.array((0., 60., 32.)),
                                np.array((180., 255., 255.)))

        self.roi_hist = cv2.calcHist([self.hsv_roi],
                                     [0], self.mask, [180], [0, 180])

        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.track_window = track_window

        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        self.id = str(uuid.uuid4)

    def track_callback(self, data):
        frame = CvBridge().imgmsg_to_cv2(data, 'bgr8')
        while time.time() < self.timeout:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, self.track_window = cv2.meanShift(dst,
                                                   self.track_window,
                                                   self.term_crit)

            # Draw it on image
            x, y, w, h = self.track_window
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

            cv2.imshow('Tracker {}'.format(self.id), frame)
            cv2.waitKey(25)

        cv2.destroyWindow('tracked object')
