from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from queue import Queue

if __name__ != "__main__":
    from . import ObjClass
    from . import userio
else:
    import ObjClass
    import userio

CACHE_EXT = ".npz"

IMAGE_FOLDER = "raw_img/"
LABEL_FOLDER = "label/"

LOAD_N_IMAGES_AT_A_TIME = 10

class DataFeeder(object):
    def __init__(self, data_path, filename="cache", dynamic_load=False,
                 data_padding=0, label_width=64, label_height=None):
        if label_height is None:
            label_height = label_width

        self.counter = 0
        self.data_path = data_path
        self.dynamic_load = dynamic_load
        self.filename = filename
        self.data_padding = data_padding

        self.label_width = label_width
        self.label_height = label_height

        self.img_data = None
        self.label_data = None

        self.file_list = None
        self.dynamic_load_buffer = Queue()

        if not dynamic_load:
            self._load_cache()
            print("DataFeeder loaded the cache")
        else:
            self.file_list = self._check_data_dir()


    def _load_cache(self):
        while True:
            cache_file = None
            try:
                cache_path = join(self.data_path, self.filename+CACHE_EXT)
                cache_file = open(cache_path, "rb")
                npz_file = np.load(cache_file)
                self.img_data = npz_file['img_data']
                self.label_data = npz_file['label_data']
                return
            except IOError as error:
                if(error.errno == 2 # No such file or directory
                   and userio.confirm("Cache not found, want to rebuild cache?")):
                    self._build_cache()
                    continue
                else: raise error
            finally:
                if cache_file is not None:
                    cache_file.close()

    def _build_cache(self):
        "Creates cache file"
        print("Building Cache")
        # Cache files
        cache_path = join(self.data_path, self.filename+CACHE_EXT)
        img_data, label_data = self._load_data()
        cache_file = open(cache_path, "wb")
        np.savez_compressed(cache_file, img_data=img_data, label_data=label_data)
        cache_file.close()
        print("Cache file created at {}".format(cache_path))

    def _check_data_dir(self):
        # Load data
        raw_img_folder_path = join(self.data_path, IMAGE_FOLDER)
        label_folder_path = join(self.data_path, LABEL_FOLDER)
        img_list = DataFeeder.get_file_list(raw_img_folder_path)
        label_list = DataFeeder.get_file_list(label_folder_path)

        # Check Files in Image and Label folder matches
        assert img_list == label_list, "Files in {} and {} do not match".format(
            raw_img_folder_path, label_folder_path)

        return (img_list)

    def _load_data(self):
        "Loads data from the provided data_path"

        file_list = self._check_data_dir()

        img_data = [None]*len(file_list)
        label_data = [None]*len(file_list)

        for file_enum in enumerate(file_list):
            counter = file_enum[0]
            file_name = file_enum[1]

            img_file_path = join(raw_img_folder_path, file_name)
            img_data[counter] = cv2.imread(img_file_path)

            label_file_path = join(label_folder_path, file_name)
            label_data[counter] = cv2.imread(label_file_path, cv2.IMREAD_UNCHANGED)

        img_data = np.array(img_data)
        label_data = np.array(label_data)
        img_data, label_data = self._breakdown_n_filter(img_data, label_data)

        p_label_data = [None]*len(label_data)
        for file_enum in enumerate(label_data):
            index = file_enum[0]
            p_label_data[index] = ObjClass.process_label(label_data[index])

        return (img_data, p_label_data)

    def _breakdown_n_filter(self, img_data, label_data):
        assert img_data.shape[0] == label_data.shape[0], "img_data, label_data count do not match"
        assert img_data.shape[1] == label_data.shape[1], "img_data, label_data height do not match"
        assert img_data.shape[2] == label_data.shape[2], "img_data, label_data width do not match"
        height = img_data.shape[1]
        width = img_data.shape[2]
        padding_y = self.data_padding
        padding_x = self.data_padding
        # print("self.data_padding", self.data_padding)

        split_label_height = self.label_height
        split_label_width = self.label_width
        # print("split_label_height", split_label_height)
        # print("split_label_width", split_label_width)

        bimg_data = [None] * img_data.shape[0]
        blabel_data = [None] * label_data.shape[0]

        start_y_indexes = range(padding_y, height-split_label_height-padding_y+1, split_label_height)
        start_x_indexes = range(padding_x, width-split_label_width-padding_x+1, split_label_width)

        total_sub_img = len(start_y_indexes) * len(start_x_indexes)

        for index, (image, label) in enumerate(zip(img_data, label_data)):
            sub_image = [None] * total_sub_img
            sub_label = [None] * total_sub_img
            counter = 0
            for start_y in start_y_indexes:
                for start_x in start_x_indexes:
                    sub_image[counter] = image[start_y-padding_y:
                                               start_y+split_label_height+padding_y,
                                               start_x-padding_x:
                                               start_x+split_label_width +padding_x]
                    sub_label[counter] = label[start_y:
                                               start_y+split_label_height,
                                               start_x:
                                               start_x+split_label_width]
                    counter += 1

            bimg_data[index] = sub_image
            blabel_data[index] = sub_label

        bimg_data = np.concatenate(bimg_data)
        blabel_data = np.concatenate(blabel_data)

        # print(bimg_data.shape)
        # print(blabel_data.shape)
        keep = np.sum(blabel_data, axis=(1, 2, 3)) > 0
        # print(keep.shape)
        bimg_data = bimg_data[keep]
        blabel_data = blabel_data[keep]

        return (bimg_data, blabel_data)


    def get_batch(self, size):
        "Returns a batch of data"
        if not self.dynamic_load:
            assert size <= len(self.img_data), "Batch bigger than Data Set"

            img_batch = [None]*size
            label_batch = [None]*size

            for counter in range(size):
                img_batch[counter] = self.img_data[self.counter]
                label_batch[counter] = self.label_data[self.counter]

                self.counter = (self.counter + 1) % len(self.img_data)

            img_batch = np.array(img_batch)
            label_batch = np.array(label_batch)

            return [img_batch, label_batch]
        else:
            img_batch = [None]*size
            label_batch = [None]*size

            raw_img_folder_path = join(self.data_path, IMAGE_FOLDER)
            label_folder_path = join(self.data_path, LABEL_FOLDER)

            while self.dynamic_load_buffer.qsize() < size:

                img_data = [None]*LOAD_N_IMAGES_AT_A_TIME
                label_data = [None]*LOAD_N_IMAGES_AT_A_TIME

                for counter in range(LOAD_N_IMAGES_AT_A_TIME):
                    file_name = self.file_list[self.counter]
                    self.counter = (self.counter + 1) % len(self.file_list)

                    img_file_path = join(raw_img_folder_path, file_name)
                    img_data[counter] = cv2.imread(img_file_path)

                    label_file_path = join(label_folder_path, file_name)
                    label_data[counter] = cv2.imread(label_file_path, cv2.IMREAD_UNCHANGED)

                img_data = np.array(img_data)
                label_data = np.array(label_data)
                img_data, label_data = self._breakdown_n_filter(img_data, label_data)

                p_label_data = [None]*len(label_data)
                for file_enum in enumerate(label_data):
                    index = file_enum[0]
                    p_label_data[index] = ObjClass.process_label(label_data[index])

                for data in zip(img_data, p_label_data):
                    self.dynamic_load_buffer.put(data)

            for counter in range(size):
                img_batch[counter], label_batch[counter] = self.dynamic_load_buffer.get()

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
