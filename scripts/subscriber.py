#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


def image_color_callback(data):
    rospy.loginfo(rospy.get_caller_id() + ' /feeding image color topic')
    cv2_img = CvBridge().imgmsg_to_cv2(data, 'bgr8')
    cv2.imshow('image', cv2_img)
    cv2.waitKey(3)


def listener():

    rospy.init_node('detection', anonymous=True)

    rospy.Subscriber('/kinect2/qhd/image_color_rect', Image,
                     image_color_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
