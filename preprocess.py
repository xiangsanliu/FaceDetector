import os

import cv2

from load_data_set import resize_image, IMAGE_SIZE

if __name__ == '__main__':
    path_name = "C:/Users/xiang/Pictures/face/others16"
    out_name = "C:/Users/xiang/Pictures/face/others3"
    i = 0
    for dir_item in os.listdir(path_name):
        output_path = '%s/l%d.jpg' % (out_name, i)
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        i += 1
        print(output_path)
        image = cv2.imread(full_path)
        os.remove(full_path)
        cv2.imwrite(output_path, image)
