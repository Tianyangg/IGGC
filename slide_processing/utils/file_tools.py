# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-9-9 上午11:24
# @Author  : tiannyang
# @File    : file_tools.py
# Comments:
import io
import os
import importlib
import torch
import sys
from imp import reload
from md_Pathological.uPACrypto.crypto import Crypto

def load_module_from_disk(pyfile):
    """ load cfg

    :param pyfile: pyfile
    :return: loaded module
    """
    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    modulename, _ = os.path.splitext(basename)

    need_reload = modulename in sys.modules

    # To avoid duplicate module name with existing modules, add the specified path first.
    os.sys.path.insert(0, dirname)
    lib = importlib.import_module(modulename)
    if need_reload:
        reload(lib)
    os.sys.path.pop(0)

    return lib


def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines


def load_enc_model(path):
    """
    :param path: model path
    :return: model params
    """
    with open(path, "rb") as fid:
        buffer = io.BytesIO(fid.read())
        buffer_value = buffer.getvalue()

        if buffer_value[0:9] == b"uAI_model":
            crypto_handle = Crypto()
            decrypt_buffer = io.BytesIO(crypto_handle.bytes_decrypt(buffer_value[128::]))
        else:
            decrypt_buffer = buffer
    params = torch.load(decrypt_buffer)
    return params
