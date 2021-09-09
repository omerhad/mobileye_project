from model import part1
from model.part2_1_cut_picture import send_images_from_array
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from model.SFM_standAlone import intagration_tfl_part4
#
import matplotlib.pyplot as plt
PATH_PLS = r'C:\networks\work\mobileye part4_integration\controller\play_list.pls'
f = open(PATH_PLS, 'r')
lst_pls_path = []
for path in f:
    lst_pls_path.append(path)
f.close()
PKL_PATH = lst_pls_path[0]
PKL_PATH = PKL_PATH[:-1]
IMAGE_NUM = lst_pls_path[1]
LST_PATH_IMG = lst_pls_path[2:]



class TFL_Man:

    def __init__(self, path_img: str, img_num: int):
        self._path = path_img
        self._num = img_num
        self._lst_candidate = part1.main(self._path)

    def save_and_crop_img(self):
        return send_images_from_array(np.array(Image.open("C:\\networks\\work\\mobileye part4_integration\\"+self._path)), self._lst_candidate)

    def chek_distance(self, array_candidate: dict):
        for i, path in enumerate(LST_PATH_IMG):
            if eval(IMAGE_NUM) + i == 29:
                break
            intagration_tfl_part4("C:\\networks\\work\\mobileye part4_integration\\"+LST_PATH_IMG[i][:-1], "C:\\networks\\work\\mobileye part4_integration\\"+LST_PATH_IMG[i + 1][:-1], "C:\\networks\\work\\mobileye part4_integration\\"+PKL_PATH, eval(IMAGE_NUM) + i,
                                                                        eval(IMAGE_NUM) + i + 1, array_candidate)
#

def controler():
    TLF_all_frame = {}
    for i, path in enumerate(LST_PATH_IMG):
        num_of_picture = eval(IMAGE_NUM) + i
        tfl_man = TFL_Man(path[:-1], num_of_picture)
        array_img, lst_pixels = tfl_man.save_and_crop_img()
        loaded_model = load_model("../model/model_20.h5")
        accuracy = []
        for i, img in enumerate(array_img):
            x = loaded_model.predict(img)[0][1]
            if x >= 0.8:
                accuracy.append(list(lst_pixels[i]))
        TLF_all_frame[num_of_picture] = accuracy
        # f.chek_distance(accuracy)
        print(TLF_all_frame)
    tfl_man.chek_distance(TLF_all_frame)


