# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-9-8 下午4:10
# @Author  : tiannyang
# @File    : WSI_svs.py
# Comments :
import numpy as np
import os
import openslide
import sys
sys.path.append("/home/tianyang/")
from PIL import Image
from utils.wsi_tile import tile_by_size

class WSI_svs:
    """ stroring information """
    def __init__(self):
        # assert isinstance(slide, openslide.OpenSlide)
        self.slide = None  # openslide.OpenSlide
        self.size = None
        self.dim_ind = None
        self.level_count = None
        # self._retreive_info(zoom=?zoom)

    def read_image(self, path):
        self.slide = openslide.open_slide(path)
        self.size = self.slide.level_dimensions[0]


    # def _retreive_info(self, zoom):
    #     """ Get 20X information from openslide"""
    #     if zoom == 20:
    #         self.dim_ind = self.level_count - 3
    #         self.size = self.slide.level_dimensions[self.dim_ind]

    def read_region(self, start_ind, level, tile_size):
        """

        :param start_ind: starting index (x, y)
        :param level: wsi level, integer
        :param tile_size: tile size (x_size, y_size)
        :return: tiled im, numpy array
        """
        tiled_im = self.slide.read_region(start_ind, level, tile_size)
        tiled_im = tiled_im.convert('RGB')
        return tiled_im


def process_single_case():
    ss = "/data/tianyang/patho_cryo_202401/prospective/wsi/WN21-02816/HS00162_WN21-02816_1_D_6_20211210180839.tif"

    wsi = WSI_svs()
    print("processing {}".format(ss))
    ss_name = os.path.basename(ss).replace(".tif", "")
    output_path = os.path.join("/data/tianyang/patho_cryo_202401/prospective/tiled_tif_redo/WN21-02816/", ss_name)

    wsi.read_image(ss)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ds = 2

    index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=0, max_pad_pixel=0, tile_size_pixel=(2720 * ds, 1824 * ds))
    for ind in index_list:
        x, y, start_ind, end_ind = ind
        tile_im = wsi.read_region(start_ind, level=(ds - 1), tile_size=(2720, 1824))
        tile_im.save(os.path.join(output_path, "R{}_C{}_.JPG".format(int(x), int(y))))
                

def process_svs_tif():
    # process_single_case()
    import pandas as pd
    import glob
    import os

    root_folder = "/data/tianyang/patho_cryo_202401/validation/wsi/"
    # save_folder = "/data/tianyang/patho_cryo_202401/prospective/tiled/"
    save_folder = "/data/tianyang/patho_cryo_202401/validation/tiled_tif_redo/"
    subjects = sorted(os.listdir(root_folder))

    # df = pd.read_csv("/data/tianyang/patho_cryo_202401/patho_cryo_20240105.csv")
    # slides_list = df["slides"].to_list()
    res_dict = {"subID":[], "sub_path": []}
    ds = 1
    for sub in subjects:
        sub_folder = os.path.join(root_folder, sub)
        # skip processed
        sub_svs = glob.glob(os.path.join(sub_folder, "*.tif"))
        # if os.path.exists(os.path.join(save_folder, sub)):
        #     print("skipping {}".format(sub))
        #     continue

        for ss in sub_svs:
            print(ss)
            try:
                wsi = WSI_svs()
                print("processing {}".format(ss))
                ss_name = os.path.basename(ss).replace(".tif", "")
                output_path = os.path.join(save_folder, sub, ss_name)

                wsi.read_image(ss)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=0, max_pad_pixel=0, tile_size_pixel=(2720 * ds, 1824 * ds))
                # index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=50, max_pad_pixel=50, tile_size_pixel=(2720, 1824))
                
                index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=0, max_pad_pixel=0, tile_size_pixel=(1056, 1056))

                for ind in index_list:
                    x, y, start_ind, end_ind = ind
                    tile_im = wsi.read_region(start_ind, level=(ds - 1), tile_size=(1056, 1056))
                    tile_im.save(os.path.join(output_path, "R{}_C{}_.JPG".format(int(x), int(y))))
                
                res_dict["subID"].append(sub)
                res_dict["sub_path"].append(output_path)
                
                pd.DataFrame(res_dict).to_csv("/data/tianyang/patho_cryo_202401/validation_real/tiled_wsi.csv", index=False)

            except Exception as e:
                print(e)

def process_shandong_histech():
    # process_single_case()
    import pandas as pd
    import glob
    import os

    # root_folder = "/data/tianyang/patho_cryo_202401/validation/wsi/"
    # # save_folder = "/data/tianyang/patho_cryo_202401/prospective/tiled/"
    # save_folder = "/data/tianyang/patho_cryo_202401/validation/tiled_tif_redo/"
    # subjects = sorted(os.listdir(root_folder))

    df = pd.read_csv("/data/tianyang/patho_cryo_202401/shandong/slide_data.csv")
    # slides_list = df["slides"].to_list()
    ds = 1

    for id, row in df.iterrows():
        if id == 0:
            slide_path = row["slide_path"]
            output_folder = row["sub_folder"]

            wsi = WSI_svs()
            print("processing {}".format(slide_path))

            wsi.read_image(slide_path)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=0, max_pad_pixel=0, tile_size_pixel=(1056, 1056))

            for ind in index_list:
                x, y, start_ind, end_ind = ind
                tile_im = wsi.read_region(start_ind, level=(ds - 1), tile_size=(1056, 1056))
                tile_im.save(os.path.join(output_folder, "R{}_C{}_.JPG".format(int(x), int(y))))



def process_svs_tif_single_case():
    # process_single_case()
    import pandas as pd
    import glob
    import os

    
    # df = pd.read_csv("/data/tianyang/patho_cryo_202401/patho_cryo_20240105.csv")
    # slides_list = df["slides"].to_list()
    
    ds = 2
    wsi_path = "/data/tianyang/patho_cryo_202401/prospective/wsi/WN21-03638/HS00229_WN21-03638_1_B_15_20211210222921.tif"
    save_folder = "/data/tianyang/patho_cryo_202401/prospective/paper_visualize/WN21-03638"

    wsi = WSI_svs()
    output_path = save_folder

    wsi.read_image(wsi_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=0, max_pad_pixel=0, tile_size_pixel=(2720 * ds, 1824 * ds))
    # index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=50, max_pad_pixel=50, tile_size_pixel=(2720, 1824))
    
    index_list = tile_by_size(ind_min=(0, 0), ind_max=wsi.size, min_pad_pixel=0, max_pad_pixel=0, tile_size_pixel=(1056*ds, 1056*ds))

    for ind in index_list:
        x, y, start_ind, end_ind = ind
        tile_im = wsi.read_region(start_ind, level=(ds - 1), tile_size=(1056, 1056))
        tile_im.save(os.path.join(output_path, "R{}_C{}_.JPG".format(int(x), int(y))))
    
      


if __name__ == "__main__":
    process_svs_tif_single_case()

   
                