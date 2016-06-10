from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

if __name__ != "__main__":
    from . import ObjClass
    from . import userio
else:
    import ObjClass
    import userio

CACHE_FILENAME = "cache.npz"

IMAGE_FOLDER = "raw_img/"
LABEL_FOLDER = "label/"

class DataFeeder(object):
    def __init__(self, data_path, data_padding=0, data_split_factor=8):
        self.counter = 0
        self.data_path = data_path
        self.data_padding = data_padding
        self.data_split_factor = data_split_factor

        self.img_data = None
        self.label_data = None

        self._load_cache()

        print("DataFeeder loaded the cache")

    def _load_cache(self):
        while True:
            cache_file = None
            try:
                cache_path = join(self.data_path, CACHE_FILENAME)
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
        cache_path = join(self.data_path, CACHE_FILENAME)
        img_data, label_data = self._load_data()
        cache_file = open(cache_path, "wb")
        np.savez_compressed(cache_file, img_data=img_data, label_data=label_data)
        cache_file.close()
        print("Cache file created at {}".format(cache_path))

    def _load_data(self):
        "Loads data from the provided data_path"
        # Load data
        raw_img_folder_path = join(self.data_path, IMAGE_FOLDER)
        label_folder_path = join(self.data_path, LABEL_FOLDER)
        img_list = DataFeeder.get_file_list(raw_img_folder_path)
        label_list = DataFeeder.get_file_list(label_folder_path)

        # Check Files in Image and Label folder matches
        assert img_list == label_list, "Files in {} and {} do not match".format(
            raw_img_folder_path, label_folder_path)

        img_data = [None]*len(img_list)
        label_data = [None]*len(img_list)

        for file_enum in enumerate(img_list):
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
        split_y = self.data_split_factor
        split_x = self.data_split_factor

        split_label_height = int((height - 2*padding_y) / split_y)
        split_label_width = int((width - 2*padding_x) / split_x)

        bimg_data = [None] * img_data.shape[0]
        blabel_data = [None] * label_data.shape[0]

        for index, (image, label) in enumerate(zip(img_data, label_data)):
            sub_image = [None] * split_y * split_x
            sub_label = [None] * split_y * split_x
            counter = 0
            for start_y in range(padding_y, height-padding_y, split_label_height):
                for start_x in range(padding_x, width-padding_x, split_label_width):
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

        return (bimg_data, blabel_data)


    def get_batch(self, size):
        "Returns a batch of data"
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
