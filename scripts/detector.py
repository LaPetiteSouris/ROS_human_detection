#!/usr/bin/env python -W ignore::DeprecationWarning
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import message_filters
from sklearn.cluster import MeanShift, estimate_bandwidth
import warnings
import uuid
from collections import namedtuple
from tracker import Tracker

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# limit above which we launch new tracker
overlap_limit = 20

background = None
fgbg = cv2.BackgroundSubtractorMOG()
full_body_cascade = './haarcascade_fullbody.xml'
human_cascade = cv2.CascadeClassifier(full_body_cascade)
tracking_list = []

bridge = CvBridge()


def detect_human(img):
    return human_cascade.detectMultiScale(img, 1.1, 1)


def get_depth_from_img(depth_img, pos):
    # TODO: verify y, x coordinate in image
    # may be it is inversed
    return depth_img[pos[1], pos[0]]


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


def cluster_contour(contours, depth_img):
    suitable_contours = []

    contour_center = None
    labels = None

    for contour in contours:
        # pass if contour is too small
        if cv2.contourArea(contour) < 50:
            continue

        suitable_contours.append(contour)

        (x, y, _, _) = cv2.boundingRect(contour)

        z = get_depth_from_img(depth_img, (x, y))

        if contour_center is not None:
            contour_center = np.vstack([contour_center, [x, y, z]])
        else:
            contour_center = np.array(([x, y, z]))

    try:
        bandwidth = estimate_bandwidth(
            contour_center, quantile=0.2, n_samples=500)

        cluster = MeanShift(bandwidth, bin_seeding=True)
        cluster.fit(contour_center)

        labels = cluster.labels_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        # print('Number of cluster: %d', n_clusters_)

    except (ValueError, AttributeError) as error:
        pass

    return suitable_contours, labels


def draw_box_around_ROI(contours, img):
    """ Draw on rectangle box for each contour cluster

    """
    # Keep track of min x, max x, min y , maxy
    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)

    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 51, 51), 2)
    cv2.putText(img, "Movement", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 127, 255), 2)

    return img[min_y:max_y, min_x:max_x], (min_x, min_y, max_x, max_y)


def draw_box_around_object(contour, gray_img, cv_img):
    """Compute the bounding box for the contour, draw it on the frame,

    """
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 51, 51), 2)
    return gray_img[y:y + h, x:x + w]


def draw_box_around_human(img, human, coordinate_origin):
    for (x, y, w, h) in human:

        # convert local coordinate to global img coordinate
        x_min_world = coordinate_origin[0]
        y_min_world = coordinate_origin[1]
        x_max_world = coordinate_origin[2]
        y_max_world = coordinate_origin[3]

        cv2.rectangle(img, (x_min_world, y_min_world),
                      (x_max_world, y_max_world), (0, 0, 255), 2)

        print('Human detected. Cooridate: x = {}, y={}'.format((
            x_min_world + x_max_world) * 0.5, (y_min_world + y_max_world) *
                                                               0.5))

        cv2.putText(img, "Hi human!", (x_max_world, y_max_world),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return img[y_min_world:y_max_world, x_min_world:
                   x_max_world], coordinate_origin


def classify_human(cv2_img, gray_img, coordinate_origin):
    human = detect_human(gray_img)

    try:
        human_roi, track_window = draw_box_around_human(cv2_img, human,
                                                        coordinate_origin)
        return human_roi, track_window
    except TypeError as e:
        pass


def calculate_overlapping_reg(rect1, rect2):
    '''

    Args:
        rect1(tuple): x, y, w, h
        rect2(tuple): x, y, w, h
    Returns:
        area(float): overlapping area
    '''
    rectangle1 = Rectangle(rect1[0], rect1[1], rect1[0] + rect1[2],
                           rect1[1] + rect1[3])

    rectangle2 = Rectangle(rect2[0], rect2[1], rect2[0] + rect2[2],
                           rect2[1] + rect2[3])
    dx = min(rectangle1.xmax, rectangle2.xmax) - max(rectangle1.xmin,
                                                     rectangle2.xmin)
    dy = min(rectangle1.ymax, rectangle2.ymax) - max(rectangle1.ymin,
                                                     rectangle2.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy

    return 0


def is_tracked(rect):
    ''' Verify if the object has been tracked

    Args:
        rect(tupple): detected rect
    Returns:
        res(boolean): True or False
    '''
    for tracker in tracking_list:
        # area = w * h
        tracked_area = tracker.track_window[2] * tracker.track_window[3]
        overlapping_percent = float(
            calculate_overlapping_reg(rect, tracker.track_window)) / float(
                tracked_area) * 100

        if overlapping_percent > overlap_limit:

            return True

        return False


def detection_callback(color_img, depth_img):

    # Convert imgs to opencv format
    color_cv2_img = CvBridge().imgmsg_to_cv2(color_img, 'bgr8')
    cv2_depth = bridge.imgmsg_to_cv2(depth_img)
    depth_array = np.array(cv2_depth, dtype=np.float32)

    # Launch main algo to start detection/tracking
    launch_detection(color_cv2_img, depth_array, color_img)


def launch_detection(color_cv2_img, depth_img, raw_data):

    # TODO: This callback does too much work, should separate
    global tracking_list

    for tracker in tracking_list:
        tracker.track_callback(raw_data, depth_img)
    '''
    try:
        tracking_list[1].track_callback(data)
        tracking_list[3].track_callback(data)
    except IndexError as error:
        pass
'''

    gray_img = process_raw_img(color_cv2_img)
    # Presume that the first frame is the background
    global background

    if background is None:
        background = gray_img

    # Foreground mask using MOG
    img_delta = fgbg.apply(gray_img)

    contours = find_contours(img_delta)

    if contours:
        cnts, labels = cluster_contour(contours, depth_img)
        # group contours by cluster

        if labels is not None:
            # Get all available cluster label
            label_unique = np.unique(labels)
            for label in label_unique:

                label_list = labels.tolist()
                idx = [i for i, x in enumerate(label_list) if x == label]

                # Draw rectangle box for each contour cluster
                object_img, coordinate_origin = draw_box_around_ROI(
                    [cnts[i] for i in idx], color_cv2_img)
        else:
            object_img, coordinate_origin = draw_box_around_ROI(contours,
                                                                color_cv2_img)
        try:
            human_roi, track_window = classify_human(color_cv2_img, object_img,
                                                     coordinate_origin)
            if human_roi is not None:
                x_min, y_min, x_max, y_max = coordinate_origin
                track_window = (x_min, y_min, x_max - x_min, y_max - y_min)

                # TODO: Check if there is an overlapping region before add
                # tracker to list
                if not is_tracked(track_window):
                    tracker = Tracker(color_cv2_img, human_roi, track_window,
                                      str(uuid.uuid4()))
                '''
                if len(tracking_list) > 5:

                    # TODO: Empty tracking list
                    tracking_list = []
                    cv2.destroyAllWindows()
                '''
                tracking_list.append(tracker)

        except TypeError as e:
            pass

    # cv2.imshow('kinect_img_background_mask', img_delta)
    cv2.imshow('kinect_img_color_detection_feed', color_cv2_img)
    cv2.waitKey(25)


def listener():

    rospy.init_node('detection', anonymous=True)

    # Subcribe multiple topics at the same time with sync
    color_sub = message_filters.Subscriber('/kinect2/qhd/image_color_rect',
                                           Image)
    depth_sub = message_filters.Subscriber('/kinect2/qhd/image_depth_rect',
                                           Image)

    ts = message_filters.TimeSynchronizer([color_sub, depth_sub], 10)

    ts.registerCallback(detection_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
