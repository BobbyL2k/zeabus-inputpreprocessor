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
    range_img = [None, None]

    range_img[0] = cv2.inRange(cv_mat_raw_label,
                               np.array([0, 0, 0, 128]),
                               np.array([255, 255, 255, 255]))

    range_img[1] = 255 - range_img[0]

    for index in range(2):
        range_img[index] = range_img[index].astype(np.float32)/255.

        range_img[index] = np.reshape(range_img[index], range_img[index].shape+(1,))

        # print(range_img[index].shape)

    return np.concatenate(range_img, axis=2)

def combine_label(cv_mat_pro_label):
    "Take in a raw label and return a processed label"
    # print(cv_mat_pro_label.shape)
    result_img = (cv_mat_pro_label[:, :, 0] > cv_mat_pro_label[:, :, 1])*255
    return result_img.astype(np.uint8)

def main():
    "Testing"
    img_mat = cv2.imread("canvas.png", cv2.IMREAD_UNCHANGED)
    # img_mat = cv2.imread(
    #     "data/label/626b7164d59e1b2a2b3cc4283c80952a5923973f.1200.png", cv2.IMREAD_UNCHANGED)

    cv2.imshow("original", img_mat)

    pro_img = process_label(img_mat)

    depro_img = combine_label(pro_img)

    print((depro_img==255).sum(), depro_img.shape)
    print((depro_img==0).sum(), depro_img.shape)
    print(((depro_img==0).sum() + (depro_img==255).sum()) == depro_img.shape[0] * depro_img.shape[1])

    cv2.imshow("depro", depro_img)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
