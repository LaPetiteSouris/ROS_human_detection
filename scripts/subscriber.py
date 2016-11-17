#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


background = None
fgbg = cv2.BackgroundSubtractorMOG()


def process_raw_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)
    return gray_img


def find_contours(img_delta):
    thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
    (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_larges_contour(contours):
    areas = [cv2.contourArea(c) for c in contours]
    largest_contour_index = np.argmax(areas)
    return contours[largest_contour_index]


def draw_box_around_object(contour, img):
    # compute the bounding box for the contour, draw it on the frame,
    # and update the text
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def image_color_callback(data):
    rospy.loginfo(rospy.get_caller_id() + ' /feeding image color topic')
    cv2_img = CvBridge().imgmsg_to_cv2(data, 'bgr8')

    gray_img = process_raw_img(cv2_img)

    global background

    if background is None:
        background = gray_img

    img_delta = fgbg.apply(gray_img)

    contours = find_contours(img_delta)
    if contours:
        largest_contour = find_larges_contour(contours)
        draw_box_around_object(largest_contour, cv2_img)

    cv2.imshow('image', cv2_img)
    cv2.waitKey(3)

    # if firist run, initialize firs frame to be background


def listener():

    rospy.init_node('detection', anonymous=True)

    rospy.Subscriber('/kinect2/qhd/image_color_rect', Image,
                     image_color_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
