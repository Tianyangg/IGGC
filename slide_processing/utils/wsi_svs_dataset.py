# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-9-8 下午5:37
# @Author  : tiannyang
# @File    : wsi_svs_dataset.py
# Comments :
from enum import Enum
from torch.utils.data import Dataset
# from md_Pathological.utils.wsi_tile import tile_by_size
# from md_Pathological.wsi_svs.svs_io import read_svs
# from md_Pathological.wsi_zj.zj_io import get_imlist
from PIL import Image
from PIL import ImageFile


class wsi_type(Enum):
    ZJ = 1
    SVS = 2


class SlideDataset(Dataset):
    def __init__(self, im_list, data_type, transform, zoom_factor=20):
        self.data_type = data_type
        self.transform = transform

        self.tile_list = im_list

    def _name2vec(self, file_name):
        file_name = file_name.split('/')[-1]
        name_noext = file_name.replace('.JPG', '')
        R_C = name_noext.split('_')
        row_ind = int(R_C[0].replace('R', ''))
        col_ind = int(R_C[1].replace('C', ''))
        return row_ind, col_ind

    def __getitem__(self, index):

        imgID = self.tile_list[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = Image.open(imgID)
        w, h = im.size
        if w != 2720 or h != 1824:
            im = im.crop((144, 0, 2720, 1824))
        if self.transform is not None:
            im = self.transform(im)
        x_ind, y_ind = self._name2vec(imgID)  # row, col index

        return imgID, x_ind, y_ind, im

    def __len__(self):
        return len(self.tile_list)


# class WSIDataset(Dataset):
#     def __init__(self, im_path, data_type, crop_size, min_pad, max_pad, transform, zoom_factor=20):
#         self.data_type = data_type
#         self.transform = transform
#         if data_type == wsi_type.ZJ:
#             # ZJ data list: a list of jpg names
#             self.tile_list = get_imlist(case_path=im_path, extention='JPG')  # tile list stores a list of images
#
#         elif data_type == wsi_type.SVS:
#             # im_path: /xxx/xxx/im_name.svs  --> path to a svs pathology file
#             self.wsi = read_svs(im_path, zoom=zoom_factor)
#             print('im loaded')
#             self.tile_list = tile_by_size(ind_min=[0, 0], ind_max=self.wsi.size,
#                                           min_pad_pixel=min_pad, max_pad_pixel=max_pad, tile_size_pixel=crop_size)
#         else:
#             raise ValueError
#
#     def __getitem__(self, index):
#         if self.data_type == wsi_type.ZJ:
#             imgID = self.tile_list[index]
#             ImageFile.LOAD_TRUNCATED_IMAGES = True
#             im = Image.open(imgID)
#             w, h = im.size
#             if w != 2720 or h != 1824:
#                 im = im.crop((144, 0, 2720, 1824))
#             if self.transform is not None:
#                 im = self.transform(im)
#             return im, imgID
#
#         elif self.data_type == wsi_type.SVS:
#             xi, yi, min_pixel, max_pixel = self.tile_list[index]
#             read_level = self.wsi.dim_ind
#             tile_size = (max_pixel - min_pixel).astype(int)
#             tiled_im = self.wsi.read_region(start_ind=min_pixel.astype(int),
#                                             level=read_level,
#                                             tile_size=tile_size)
#             tiled_im = self.transform(tiled_im)
#
#             return xi, yi, tiled_im
#         else:
#             raise ValueError
#
#     def __len__(self):
#         return len(self.tile_list)