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
        "Returns bgra color in numpy array of length 4"
        hex_str = self.color_hex_str
        return np.append(_hex_str_rgb_2_bgr_parser(hex_str), alpha)

    def get_bgr(self):
        "Returns bgr color in numpy array of length 3"
        hex_str = self.color_hex_str
        return _hex_str_rgb_2_bgr_parser(hex_str)

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
        index = cl_enum[0]
        range_img = cv2.inRange(cv_mat_raw_label,
                                CLASS[index].get_bgra(0),
                                CLASS[index].get_bgra(255))

        range_img = range_img.astype(np.float32)/255.

        range_imgs[index] = np.reshape(range_img, range_img.shape+(1,))

    return np.concatenate(range_imgs, axis=2)

def combine_label(cv_mat_pro_label):
    "Take in a raw label and return a processed label"
    # get image shape without the class count
    img_dimesion = cv_mat_pro_label.shape[:-1]

    range_imgs = np.rollaxis(cv_mat_pro_label, axis=2)

    lower_bound = np.array([0.5])
    upper_bound = np.array([1.0])

    result_img = np.zeros(shape=img_dimesion + (3,)).astype(np.uint8)

    for cl_enum in enumerate(CLASS):
        index = cl_enum[0]
        obj_class = cl_enum[1]
        range_img = cv2.inRange(range_imgs[index], lower_bound, upper_bound)
        # print(range_img.shape, img_dimesion + (1,))
        range_img = np.reshape(range_img, img_dimesion + (1,)) / 255
        b_color, g_color, r_color = obj_class.get_bgr()
        b_sub_img = range_img * b_color
        g_sub_img = range_img * g_color
        r_sub_img = range_img * r_color

        bgr_img = np.concatenate([b_sub_img, g_sub_img, r_sub_img], axis=2).astype(np.uint8)

        # print(result_img.dtype, bgr_img.dtype)

        result_img = cv2.add(result_img, bgr_img)

    return result_img

def main():
    "Testing"
    img_mat = cv2.imread("canvas.png", cv2.IMREAD_UNCHANGED)
    # img_mat = cv2.imread(
    #     "data/label/626b7164d59e1b2a2b3cc4283c80952a5923973f.1200.png", cv2.IMREAD_UNCHANGED)

    cv2.imshow("original", img_mat)

    pro_img = process_label(img_mat)

    depro_img = combine_label(pro_img)

    cv2.imshow("depro", depro_img)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
