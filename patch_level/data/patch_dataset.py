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

from patch_level.utils.augmentation import augmentation


def read_csv(filename, weight=True):
    """ read data file and return datapath, label sublabel"""
    df = pd.read_csv(filename)
    # df = df[0:10000]
    # im_path	label	sublabel
    imlist_old = df['im_path'].tolist()
    imlist = [x.replace("/data/tianyang/", "/mnt/data198/tianyang/") for x in imlist_old]
    label = df['label'].tolist()
    sublabel = df['sublabel'].tolist()

    # label_weight = None
    # if weight:
    #     label_weight = df['label_weight'].fillna(1)

    return imlist, label, sublabel



class PatchDataset(Dataset):
    """Designed to train patch level classification"""

    def __init__(self, imlist_file, is_train=True, target='tumour'):  # crop_size,

        self.imlist, self.label, self.sublabel = read_csv(imlist_file)
        # self.crop_size = crop_size
        # self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        # convnext transform
        # train_transform = transforms.Compose([
        #     transforms.Resize((912, 1360)),
        #     transforms.RandomCrop((896, 1344)),
        #     # transforms.CenterCrop((1536, 2274)),
        #     # transforms.Resize((456, 675)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        # test_transform = transforms.Compose([
        #     transforms.Resize((912, 1360)),
        #     transforms.CenterCrop((896, 1344)),
        #     # transforms.Resize((456, 675)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        # vit transform
        train_transform = transforms.Compose([
            transforms.Resize((1216, 1812)),
            transforms.RandomCrop((704, 704)),  # 512 512
            # transforms.CenterCrop((1536, 2274)),
            # transforms.Resize((456, 675)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((1216, 1812)),
            transforms.CenterCrop((704, 704)),
            # transforms.Resize((456, 675)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if is_train:
            self.transform = train_transform
        else:
            self.transform = test_transform

        self.is_train = is_train
        self.target = target

        if self.target == 'tumour' or self.target == "risk_grade":
            # label" Tumor, NonTumor, BG
            # sublabel: AA A AO GBM O
            self.label_dict = {"BG": 0, "Tumor": 1, "Nontumor": 2}
            self.sublabel_dict = {"O": 0, "A": 1, "AO":2, "AA": 3, "GBM": 4}
        elif self.target == "risk":
            self.risk_dict = {"low": 0, "high": 1}
        elif self.target == "celltype":
            self.risk_dict = {"low": 0, "high": 1}
            self.sublabel_dict = {"O": 0, "A": 1, "AO":2, "AA": 3, "GBM": 4}
        elif self.target == "risk_grade3_new":
            self.label_dict = {'BG': 0, 'Nontumor': 1, 'Grade2': 2, 'Grade3': 3, 'Grade4': 4}
        elif self.target == "gbm_vs_other":
            self.label_dict = {'BG': 0, 'Nontumor': 1, 'Grade2': 2, 'Grade3': 3, 'Grade4': 4}

        else:
            raise NotImplementedError
            #self.label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
            
            
    def __getitem__(self, idx):
        # imgID = self.imlist[idx]
        imgID = self.imlist[idx].replace("/data/tianyang/", "/mnt/data198/tianyang/") # "{}".format()
        # load and transform image
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = Image.open(imgID)
        
        w, h = im.size
        if w != 2720 or h != 1824:
            im = im.crop((144, 0, 2720, 1824))

        if self.is_train:
            im = augmentation(im)

        if self.transform is not None:
            im = self.transform(im)

        # process label
        patch_label = self.label[idx]
        sublabel = self.sublabel[idx]

        if self.target == "tumour":
            # train tumour, nt, BG in this situation
            label = self.label_dict[patch_label] # {"BG": 0, "Tumor": 1, "NonTumor": 2}
        elif self.target == "celltype":
            if patch_label == "BG":
                label = 0
            elif patch_label == "Nontumor":
                label = 1
            else:
                # tumour
                if sublabel in ["O2", "O3", "AO3", "AO", "O"]:
                    label = 2
                elif sublabel in ["A2", "A3", "A4", "GBM4", "AA3", "A", "AA"]:
                    label = 3
                else:
                    raise NotImplementedError
                
        elif self.target == "risk":
            # sublabel is AA AO or GBM
            if patch_label in ["BG", "Nontumor"]:
                # low risk
                label = self.risk_dict["low"]
            else: # tumour
                if sublabel in ["AA", "AO", "GBM"]:
                    label = self.risk_dict["high"]
                else:
                    label = self.risk_dict["low"]

        elif self.target == "risk_grade":
            # BG - 0; NT - 1; O/A - 2; AA - 3; AO / GBM A4
            # BG - 0; NT - 1; O/A - 2; AA AO - 3; GBM 4
            if patch_label == "BG":
                label = 0
            elif patch_label == "Nontumor":
                label = 1
            elif patch_label == "Grade2":
                label = 2
            elif patch_label == "Grade3":
                label = 3
            elif patch_label == "Grade4":
                label = 4
            else:
                raise NotImplementedError
                
        elif self.target == "risk_grade3_new":
            label = self.label_dict[patch_label]
            
        elif self.target == "gbm_vs_other":
            if patch_label == "Grade4":  # high risk
                if sublabel in ["A4", "GBM4", "GBM", "Glioblastoma"]:
                    label = 1
                else:
                    label = 0
            else:  # all other grades are treated as non gbm markers
                label = 0
        
        else:
            raise NotImplementedError

        label = torch.tensor(label).long()
        
        return im, label, imgID

    def normalization(self, im):
        ims = np.array(im)
        mean = np.mean(ims)
        std = np.std(ims)
        return (ims - mean) / std

    def __len__(self):
        return len(self.imlist)