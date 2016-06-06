"""Information about each classes"""
import cv2
import numpy as np


def _hex_str_rgb_2_bgr_parser(hex_str):
    r_str = hex_str[0:2]
    g_str = hex_str[2:4]
    b_str = hex_str[4:6]

    r_value = int(r_str, 16)
    g_value = int(g_str, 16)
    b_value = int(b_str, 16)

    return np.array([b_value, g_value, r_value])

class ObjClass(object):
    "Framework Class for Object classes"
    color_hex_str = "FFFFFF"

    def __init__(self, color_hex_str):
        self.color_hex_str = color_hex_str

    def get_bgra(self, alpha):
        "Returns rgba color in numpy array of length 4"
        hex_str = self.color_hex_str
        return np.append(_hex_str_rgb_2_bgr_parser(hex_str), alpha)

# pylint: disable=C0326
RED_BOUY   = ObjClass("F44336")
GREEN_BUY  = ObjClass("4CAF50")
YELLOW_BOY = ObjClass("FFEB3B")
GATE       = ObjClass("FF9800")
NAVIGATE   = ObjClass("795548")
PATH       = ObjClass("CDDC39")
SET_COURE  = ObjClass("E91E63")
BIN        = ObjClass("8BC34A")
COIN       = ObjClass("00BCD4")

CLASS = [
    RED_BOUY,
    GREEN_BUY,
    YELLOW_BOY,
    GATE,
    NAVIGATE,
    PATH,
    SET_COURE,
    BIN,
    COIN,
]

def process_label(cv_mat_raw_label):
    "Take in a raw label and return a processed label"
    range_imgs = [None]*len(CLASS)

    for cl_enum in enumerate(CLASS):
        range_img = cv2.inRange(cv_mat_raw_label,
                                CLASS[cl_enum[0]].get_bgra(0),
                                CLASS[cl_enum[0]].get_bgra(255))

        range_img = range_img.astype(np.float32)/255.

        range_imgs[cl_enum[0]] = np.reshape(range_img, range_img.shape+(1,))

    return np.concatenate(range_imgs, axis=2)
