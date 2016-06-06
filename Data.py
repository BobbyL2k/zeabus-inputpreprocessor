from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from . import ObjClass

class DataFeeder(object):
    def __init__(self, data_path):
        self.counter = 0
        self.path_raw_img = data_path+"raw_img/"
        self.path_raw_label = data_path+"label/"

        img_list = DataFeeder.get_file_list(self.path_raw_img)
        label_list = DataFeeder.get_file_list(self.path_raw_label)

        # Check Files in Image and Label folder matches
        assert img_list == label_list, "Files in {} and {} do not match".format(
            self.path_raw_img, self.path_raw_label)

        self.img_cache = [None]*len(img_list)
        self.label_cache = [None]*len(img_list)

        for file_enum in enumerate(img_list):
            counter = file_enum[0]
            file_name = file_enum[1]

            img_file_path = join(self.path_raw_img, file_name)
            self.img_cache[counter] = cv2.imread(img_file_path)

            label_file_path = join(self.path_raw_label, file_name)
            self.label_cache[counter] = ObjClass.process_label(
                cv2.imread(label_file_path, cv2.IMREAD_UNCHANGED))

        print("DataFeeder Cache Complete")

    def get_batch(self, size):
        "Returns a batch of data"
        assert size <= len(self.img_cache), "Batch bigger than Data Set"

        img_batch = [None]*size
        label_batch = [None]*size

        for counter in range(size):
            img_batch[counter] = self.img_cache[self.counter]
            label_batch[counter] = self.label_cache[self.counter]

            self.counter = (self.counter + 1) % len(self.img_cache)

        img_batch = np.array(img_batch)
        label_batch = np.array(label_batch)

        return [img_batch, label_batch]

    @staticmethod
    def get_file_list(path):
        "Return List of file in the provided path"
        file_list = [file_name
                     for file_name in listdir(path)
                     if isfile(join(path, file_name))]
        file_list.sort()
        return file_list

def main():
    "Example Usage"
    feeder = DataFeeder("data/")

    print("data", feeder.get_batch(10)[0].shape)
    print("label", feeder.get_batch(10)[1].shape)


if __name__ == "__main__":
    main()
