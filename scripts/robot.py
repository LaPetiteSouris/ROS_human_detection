import cv2
import numpy as np


def get_depth_from_img(depth_img, pos):
    # TODO: verify y, x coordinate in image
    # may be it is inversed
    return depth_img[pos[1], pos[0]]


def find_robot(scene, depth_img):
    needle = cv2.imread('./robot.png')
    haystack = scene

    result = cv2.matchTemplate(needle, haystack, cv2.TM_CCOEFF_NORMED)
    y, x = np.unravel_index(result.argmax(), result.shape)
    z = get_depth_from_img(depth_img, (x, y))
    return x, y, z
