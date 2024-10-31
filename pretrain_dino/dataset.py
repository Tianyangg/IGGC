import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from PIL import ImageEnhance
import torch
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import pretrain_dino.utils as utils


def read_csv(filename, weight=True):
    """ read data file and return datapath, label sublabel"""
    df = pd.read_csv(filename)
    # im_path	label	sublabel
    imlist = df['im_path'].tolist()

    return imlist



class PatchDataset(Dataset):
    """Designed to train patch level classification"""

    def __init__(self, imlist_file):  # crop_size,

        self.imlist = read_csv(imlist_file)
        
        self.transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1.), 
            local_crops_scale=(0.05, 0.4), 
            local_crops_number=8
            )
        
            
    def __getitem__(self, idx):
        # imgID = self.imlist[idx]
        imgID = self.imlist[idx].replace("/data/tianyang/", "/mnt/data198/tianyang/") # "{}".format()
        # load and transform image
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = Image.open(imgID)
        
        w, h = im.size
        if w != 2720 or h != 1824:
            im = im.crop((144, 0, 2720, 1824))
        
        im_list = self.transform(im)
        
        return im_list  # a list of tensor


    def __len__(self):
        return len(self.imlist)


class DataAugmentationDINO(object):
    """ Self defined augmentation transform"""
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.Resize([1216, 1812]),  # resize image
            transforms.RandomResizedCrop(512, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.Resize([1216, 1812]), 
            transforms.RandomResizedCrop(512, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.Resize([1216, 1812]), 
            transforms.RandomResizedCrop(128, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops