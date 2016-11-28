#!/usr/bin/env python -W ignore::DeprecationWarning
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from sklearn.cluster import MeanShift, estimate_bandwidth


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


def cluster_contour(contours):
    suitable_contours = []

    contour_center = None
    labels = None

    for contour in contours:
        # pass if contour is too small
        if cv2.contourArea(contour) < 50:
            continue

        suitable_contours.append(contour)

        (x, y, _, _) = cv2.boundingRect(contour)
        if contour_center is not None:
            contour_center = np.vstack([contour_center, [x, y]])
        else:
            contour_center = np.array(([x, y]))

    try:
        # TODO: if there is only one contour, cluster is not needed
        # Should handle case where only one contour found (e.g: labels is None)
        bandwidth = estimate_bandwidth(contour_center,
                                       quantile=0.2, n_samples=500)

        cluster = MeanShift(bandwidth, bin_seeding=True)
        cluster.fit(contour_center)

        labels = cluster.labels_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print('Number of cluster: %d', n_clusters_)

    except (ValueError, AttributeError) as error:
        pass

    return suitable_contours, labels


def find_larges_contour(contours):
    """Return largest contour in the list of contour
    """
    areas = [cv2.contourArea(c) for c in contours]
    largest_contour_index = np.argmax(areas)
    return contours[largest_contour_index]


def draw_box_around_ROI(contours, img):
    """ Draw on rectangle box for each contour cluster

    """
    # Keep track of min x, max x, min y , maxy
    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)

    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 51, 51), 2)


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

    if contours:
        cnts, labels = cluster_contour(contours)
        # group contours by cluster
        # TODO: if labels is None, there is only one contour
        # Should process that contour instead of process
        # contour by cluster

        if labels is not None:
            # Get all available cluster label
            label_unique = np.unique(labels)
            for label in label_unique:

                label_list = labels.tolist()
                idx = [i for i, x in enumerate(label_list) if x == label]

                # Draw rectangle box for each contour cluster
                draw_box_around_ROI([cnts[i] for i in idx], cv2_img)

    # TODO: instead of pass img captured inside a contour to classifier
    # We must now get the image covered by a cluster of contour
    # Then pass it to classifier

    #for cnt in contours:
        #object_img = draw_box_around_object(cnt, gray_img, cv2_img)
        #classify_human(cv2_img, object_img)

    cv2.imshow('kinect_img_background_mask', img_delta)
    cv2.imshow('kinect_img_color_detection_feed', cv2_img)
    cv2.waitKey(0)


def listener():

    rospy.init_node('detection', anonymous=True)

    rospy.Subscriber('/kinect2/qhd/image_color_rect', Image,
                     image_color_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
