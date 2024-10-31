# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-9-8 下午4:17
# @Author  : tiannyang
# @File    : svs_io.py
# Comments:

import openslide
from md_Pathological.wsi_svs.WSI_svs import WSI_svs


def read_svs(file_name, zoom=20):
    """ read an svs format WSI

    :param file_name:
    :param dtype:
    :return: an openslide.Openslide object
    """
    assert file_name.endswith(".svs")
    try:
        slide = openslide.open_slide(file_name)
    except Exception:
        raise IOError("Failed to read WSI from disk")

    wsi = WSI_svs(slide, zoom=zoom)
    return wsi