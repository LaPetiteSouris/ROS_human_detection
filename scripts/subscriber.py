#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


background = None
fgbg = cv2.BackgroundSubtractorMOG()
full_body_cascade = './haarcascade_fullbody.xml'
human_cascade = cv2.CascadeClassifier(full_body_cascade)


def detect_human(img):
    return human_cascade.detectMultiScale(img, 1.1, 1)


def process_raw_img(img):
    """ Convert BGR image to gray scale and blur for processing

    Args:
        img(cv2 image): BGR image

    Returns:
        cv2 image: gray scale image

    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    return gray_img


def find_contours(img_delta):
    """ Find all contours in image mask

    Args:
        img_delta(cv2 foreground mask):

    Returns:
        list: list of available contours
    """
    thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
    (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    return contours

# TODO: now have 2D array of x,y of all contours' centers, we muse proceed
# to clustering
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift.fit


def cluster_contour(contours):
    contour_center = None

    for contour in contours:
        (x, y, _, _) = cv2.boundingRect(contour)
        if contour_center:
            contour_center = np.vstack([contour_center, [x, y]])
        else:
            contour_center = np.array(([x, y]))

    return contour_center


def find_larges_contour(contours):
    """Return largest contour in the list of contour
    """
    areas = [cv2.contourArea(c) for c in contours]
    largest_contour_index = np.argmax(areas)
    return contours[largest_contour_index]


def draw_box_around_object(contour, gray_img, cv_img):
    """Compute the bounding box for the contour, draw it on the frame,

    """
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 51, 51), 2)
    return gray_img[y: y + h, x: x + w]


def draw_box_around_human(img, human):
    for (x, y, w, h) in human:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


def classify_human(cv2_img, gray_img):
    human = detect_human(gray_img)
    draw_box_around_human(cv2_img, human)


def image_color_callback(data):
    cv2_img = CvBridge().imgmsg_to_cv2(data, 'bgr8')

    gray_img = process_raw_img(cv2_img)
    # Presume that the first frame is the background
    global background

    if background is None:
        background = gray_img

    # Foreground mask using MOG
    img_delta = fgbg.apply(gray_img)

    contours = find_contours(img_delta)
    object_img = None

    # TODO: Analyze cluster of contour here
    if contours:
        cluster_contour(contours)

    # TODO: instead of pass img captured inside a contour to classifier
    # We must now get the image covered by a cluster of contour
    # Then pass it to classifier

    for cnt in contours:
        object_img = draw_box_around_object(cnt, gray_img, cv2_img)
        classify_human(cv2_img, object_img)

    cv2.imshow('kinect_img_background_mask', img_delta)
    cv2.imshow('kinect_img_color_detection_feed', cv2_img)
    cv2.waitKey(25)


def listener():

    rospy.init_node('detection', anonymous=True)

    rospy.Subscriber('/kinect2/qhd/image_color_rect', Image,
                     image_color_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
