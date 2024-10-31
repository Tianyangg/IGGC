# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-9-8 下午2:24
# @Author  : tiannyang
# @File    : wsi_tile.py
# Comments :
import numpy as np


class TiledWSI:
    def __init__(self):
        self.ind_x = None
        self.ind_y = None
        self.pixel_indx = None
        self.pixel_indy = None

    def assign(self, indx, indy, pixelx, pixely):
        self.ind_x = indx
        self.ind_y = indy
        self.pixel_indx = pixelx
        self.pixel_indy = pixely

    def get(self):
        return self.ind_x, self.ind_y, self.pixel_indx, self.pixel_indy


def tile_by_size(ind_min, ind_max, min_pad_pixel, max_pad_pixel, tile_size_pixel):
    """ Tile a WSI into fixed size patches

    :param ind_min: [x, y], index start to tile
    :param ind_max: [x, y], index end tiling
    :param min_pad_pixel: overlap pixel along the minimum edges (left vertical and top horizontal)
    :param max_pad_pixel: overlap pixel along the maximun edges (right vertical and bottom horizontal)
    :param tile_size_pixel: tiled image size
    :return: a list of grid index and pixel index, each with 4 entity
    """
    assert len(tile_size_pixel) == 2  # [x, y] -- pixel number

    ind_min = np.array(ind_min, dtype=np.float32)
    ind_max = np.array(ind_max, dtype=np.float32)
    min_pad_pixel = np.array(min_pad_pixel, dtype=np.float32)
    max_pad_pixel = np.array(max_pad_pixel, dtype=np.float32)
    tile_size_pixel = np.array(tile_size_pixel, dtype=np.float32)

    real_tile_size = tile_size_pixel + min_pad_pixel + max_pad_pixel  # added overlap

    # number of blocks along each dim
    block_nums = np.ceil((ind_max - ind_min) / tile_size_pixel).astype(np.float32)

    total = np.prod(block_nums)
    ind_list = []

    for idx in range(int(total)):
        yi = idx % block_nums[1]
        xi = idx // block_nums[1]
        coord = np.array((xi, yi), dtype=np.float32)

        # compute tile min and max index
        out_min_pixel = ind_min + coord * tile_size_pixel
        out_max_pixel = out_min_pixel + tile_size_pixel

        # if one dimension exceeds border, recompute index
        # out_min_pixel -= min_pad_pixel
        # out_max_pixel += max_pad_pixel

        for j in range(2):  # x and y
            if out_max_pixel[j] > ind_max[j]:
                out_max_pixel[j] = ind_max[j]
                out_min_pixel[j] = out_max_pixel[j] - real_tile_size[j]

            if out_min_pixel[j] < out_min_pixel[j]:
                out_min_pixel[j] = ind_min[j]
                out_max_pixel[j] = out_min_pixel[j] + real_tile_size[j]

        ind_list.append([xi, yi, out_min_pixel, out_max_pixel])

    return ind_list