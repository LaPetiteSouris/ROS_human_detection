#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial import distance
import notify2


class Tracker:
    def __init__(self, frame, roi, track_window, uuid, robot_pos):

        self.hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv_roi,
                                np.array((0., 60., 32.)),
                                np.array((180., 255., 255.)))

        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.mask, [180],
                                     [0, 180])

        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.track_window = track_window
        self.id = uuid
        self.paths = []
        # Caliberation factor: 1 px =5.6mm
        self.calib = 5.6
        self.robot_pos = (robot_pos[0] * self.calib, robot_pos[1] * self.calib,
                          robot_pos[2])

        # Setup the termination criteria, either 10 iteration or move by at
        # least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                          1)
        self.id = uuid

        # Distance
        self.distance = []
        notify2.init(self.id)

    def get_depth_from_img(self, depth_img, pos):
        return depth_img[pos[1], pos[0]]

    def calculate_distance_to_robot(self, obj_pos):
        return distance.euclidean(self.robot_pos, obj_pos)

    def moving_average(self, interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    def get_alert_condition(self):
        # Lastest distance
        dis = self.distance[-1]
        if dis < 1500 and dis > 1000:
            return 'Warning: Agent is too close with dangerous zone !', (
                0, 255, 255)

        elif dis < 1000 and dis > 600:
            return 'Alert: Agent is inside dangerous zone !', (0, 64, 255)

        elif dis < 600 and dis > 0:
            return 'Critical: Agent is in collision with robot !', (0, 0, 255)

        return None, None

    def track_callback(self, data, depth_img):
        frame = CvBridge().imgmsg_to_cv2(data, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, self.track_window = cv2.CamShift(dst, self.track_window,
                                              self.term_crit)

        # Draw it on image

        x, y, w, h = self.track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        z = self.get_depth_from_img(depth_img, (x, y))
        cor_text = 'x={}, y={},z={}'.format(x * self.calib, y * self.calib, z)

        cv2.putText(frame, cor_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        self.paths.append((x * self.calib, y * self.calib, z))

        distance = self.calculate_distance_to_robot((x * self.calib,
                                                     y * self.calib, z))

        print('Distance between object and forbidden zone is {} mm'.format(
            distance))

        self.distance.append(distance)

        # alert sitution check
        status_alert, color_text = self.get_alert_condition()

        # insert warning/alert into frame
        if status_alert:
            cv2.putText(frame, status_alert, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 2)

        cv2.imshow('Tracker {}'.format(self.id), frame)
        cv2.waitKey(33)
