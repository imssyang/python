from copy import deepcopy
import json
import numpy as np
import cv2 as cv


VERSION = cv.__version__
BLUE = [255, 0, 0]
GREEN = [0, 255, 0]
RED = [0, 0, 255]


class CVImg:
    def __init__(self, path):
        self.img = cv.imread(path, cv.IMREAD_COLOR)
        self.width, self.height, self.channel = self.__shape(self.img)
        self.pixel_size = self.img.size
        self.data_type = str(self.img.dtype)
        self.__copy_roi(self.img)
        self.__set_red_channle(self.img)
        self.__opeator_channel(self.img)
        self.img = self.__make_border(self.img)
        self.__color_space_flags()
        self.mask_img, self.obj_img = self.__get_blue_object(self.img)
        self.hsv_blug = self.__get_hsv_color(BLUE)
        self.hsv_green = self.__get_hsv_color(GREEN)
        self.hsv_red = self.__get_hsv_color(RED)

    def __shape(self, img):
        if len(img.shape) == 2:
            width, height = img.shape
            channel = 1
        else:
            width, height, channel = img.shape
        return width, height, channel

    def __copy_roi(self, img):
        roi = img[10:30, 10:30]
        img[10:30, 50:70] = roi

    def __set_red_channle(self, img):
        pass  # img[:,:,2] = 123

    def __opeator_channel(self, img):
        b, g, r = cv.split(img)
        img = cv.merge((b, g, r))

    def __make_border(self, img):
        return cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)

    def __get_hsv_color(self, bgr):
        g = np.uint8([[bgr]])
        h = cv.cvtColor(g, cv.COLOR_BGR2HSV)
        return h.tolist()

    def __color_space_flags(self):
        return [i for i in dir(cv) if i.startswith("COLOR_")]

    def __get_blue_object(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(img, img, mask=mask)
        return mask, res

    @property
    def json(self):
        d = deepcopy(self.__dict__)
        for attr in ["img", "mask_img", "obj_img"]:
            d.pop(attr)
        return json.dumps(d, indent=4)

    def show(self, msec: int = 3000):
        print(self.json)
        if self.img is not None:
            cv.imshow("origin", self.img)
        if self.mask_img is not None:
            cv.imshow("mask", self.mask_img)
        if self.obj_img is not None:
            cv.imshow("object", self.obj_img)
        k = cv.waitKey(msec)


if __name__ == "__main__":
    print(VERSION)
    ci = CVImg("1.jpg")
    ci.show()
