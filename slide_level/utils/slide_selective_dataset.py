import torch
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.im_augmentation import augmentation
import random
Image.LOAD_TRUNCATED_IMAGES = True


class SelectiveDatasetMIL(Dataset):
    def __init__(self, train_csv, target="glioma", select_number=20, istrain=True):
        """Assume train_csv with three colums, subID, image_path, feature_path"""
        self.df_train = pd.read_csv(train_csv)
        self.subject_list = self.df_train["subID"]
        self.feature_path_list = self.df_train["feature_path"]
        self.subject_label_list = self.df_train["subLabel"]
        self.target = target
        self.select_number = select_number + 1 
        self.istrain = istrain
        self.bg_label = [0]
        if self.target == "glioma":
            self.tumour_label = [1, 2]
            self.non_tumour_label = [3]
            self.selective_features = ["Glioma", "MVPNetwork"]
            self.label_dict = {"O": 0, "A": 1, "AO": 2, "AA": 3, "GBM": 4}

        elif self.target == "toy":
            self.tumour_label = [1, 2]
            self.non_tumour_label = [3]
            self.selective_features = ["Glioma"]
            self.label_dict = {"O": 0, "A": 1, "AO": 2, "AA": 3, "GBM": 4}

        elif self.target == "cns_tumour":
            # BG O A NT MET LYM EPE
            self.tumour_label = [1, 2, 4, 5, 6]
            self.non_tumour_label = [3]
            self.selective_features = ["SevenTumor"]
            self.label_dict = {"O": 0, "A": 1, "AO": 0, "AA": 1, "GBM": 1, "NT": 2, "MET": 3, "LYM": 4, "EPE": 5, "AEPE": 5}
            
        
        self.transform = transforms.Compose([
                transforms.CenterCrop((1536, 2274)),
                transforms.Resize((456, 675)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
               
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        
    def __len__(self):
        """must implement"""
        return len(self.subject_list)
    
    def _select_image_case_celltype(self, celltype_csv, select_num=10, use_prob=False):
        """{O A AO AA GBM} {BG, O, A, NT}

        Args:
            celltype_csv (_type_): _description_
            select_num (int, optional): _description_. Defaults to 10.
            use_prob (bool, optional): where to use probability to determine patches Defaults to False.

        Returns:
            _type_: _description_
        """
        
        bk_tumour_prd_labels = self.tumour_label + self.non_tumour_label
        tumour_prd_labels = self.tumour_label
        # randomly select patches
        real_select_num = select_num + 1  # for backup
        df_celltype = pd.read_csv(celltype_csv)

        candidate_dict_celltype = {"images": [], "celltype": []}
        candidate_dict_celltype_bk = {"images": [], "celltype": []}

       
        # Union tumour region
        ctr = 0
        prob_threshold = 0.6
        if use_prob:
            # assert "prob" in df_celltype.headers()
            for id, row in df_celltype.iterrows():
                # if celltype is tumour, append to candidate 
                celltype_prd = row["prd"]
                celltype_prob = row["prob"]
                if celltype_prd in tumour_prd_labels and celltype_prob > prob_threshold:
                    candidate_dict_celltype["images"].append(row["images"])
                    candidate_dict_celltype["celltype"].append(row["prd"])

                if celltype_prd in bk_tumour_prd_labels:
                    candidate_dict_celltype_bk["images"].append(row["images"])
                    candidate_dict_celltype_bk["celltype"].append(row["prd"])

            # update the number to select
            real_select_num = int(0.3 * len(candidate_dict_celltype))

        else:
            for id, row in df_celltype.iterrows():
                # if celltype is tumour, append to candidate 
                celltype_prd = row["prd"]
                if celltype_prd in tumour_prd_labels:
                    candidate_dict_celltype["images"].append(row["images"])
                    candidate_dict_celltype["celltype"].append(row["prd"])

                if celltype_prd in bk_tumour_prd_labels:
                    candidate_dict_celltype_bk["images"].append(row["images"])
                    candidate_dict_celltype_bk["celltype"].append(row["prd"])
                
        # sample image
        if len(candidate_dict_celltype["celltype"]) > real_select_num:
            select_cell = np.random.choice(candidate_dict_celltype["images"], real_select_num, replace=False)
        elif len(candidate_dict_celltype["celltype"]) > 0:
            select_cell = np.random.choice(candidate_dict_celltype["images"], real_select_num, replace=True)
        else:
            if len(candidate_dict_celltype_bk["celltype"]) > real_select_num:
                select_cell = np.random.choice(candidate_dict_celltype_bk["images"], real_select_num, replace=False)
            elif len(candidate_dict_celltype_bk["celltype"]) > 0:
                select_cell = np.random.choice(candidate_dict_celltype_bk["images"], real_select_num, replace=True)
        
        selected_im = select_cell
        return selected_im
    
    def _select_image_case(self, celltype_csv, risk_csv, select_num=10, use_prob=False):
        """
        training target: {O A AO AA GBM} 
        patch label {BG: 0, O: 1, A: 2, NT: 3}

        Args:
            celltype_csv (_type_): _description_
            risk_csv (_type_): _description_
            select_num (int, optional): _description_. Defaults to 10.
            use_prob (bool, optional): _description_. where to use probability to determine patches Defaults to False.

        Returns:
            _type_: _description_
        """
        
        bk_tumour_prd_labels = [1, 2]
        tumour_prd_labels = [1, 2]
        risk_label_list = [1]
        # randomly select patches
        real_select_num = select_num + 1  # for backup
        df_celltype = pd.read_csv(celltype_csv)
        df_risk = pd.read_csv(risk_csv)

        candidate_dict_celltype = {"images": [], "celltype": []}
        candidate_dict_celltype_bk = {"images": [], "celltype": []}
        candidate_dict_risk = {"images": [], "risk": []}
        # Union tumour region
        ctr = 0

        prob_threshold = 0.6
        if use_prob:
            # assert "prob" in df_celltype.headers()
            for id, row in df_celltype.iterrows():
                # if celltype is tumour, append to candidate 
                celltype_prd = row["prd"]
                celltype_prob = row["prob"]
                if celltype_prd in tumour_prd_labels and celltype_prob > prob_threshold:
                    candidate_dict_celltype["images"].append(row["images"])
                    candidate_dict_celltype["celltype"].append(row["prd"])

                    risk = df_risk.loc[ctr, "prd"]
                    if risk in risk_label_list:
                        candidate_dict_risk["images"].append(row["images"])
                        candidate_dict_risk["risk"].append(risk)

                if celltype_prd in bk_tumour_prd_labels:
                    candidate_dict_celltype_bk["images"].append(row["images"])
                    candidate_dict_celltype_bk["celltype"].append(row["prd"])

            # update the number to select
            real_select_num = int(0.3 * len(candidate_dict_celltype))
        
        for id, row in df_celltype.iterrows():
            # if celltype is tumour, append to candidate 
            celltype_prd = row["prd"]
            if celltype_prd in tumour_prd_labels:
                candidate_dict_celltype["images"].append(row["images"])
                candidate_dict_celltype["celltype"].append(row["prd"])
                
                risk = df_risk.loc[ctr, "prd"]
                if risk in risk_label_list:
                    candidate_dict_risk["images"].append(row["images"])
                    candidate_dict_risk["risk"].append(risk)

            if celltype_prd in bk_tumour_prd_labels:
                candidate_dict_celltype_bk["images"].append(row["images"])
                candidate_dict_celltype_bk["celltype"].append(row["prd"])
                
        # choose image
        if len(candidate_dict_celltype["celltype"]) > real_select_num:
            select_cell = np.random.choice(candidate_dict_celltype["images"], real_select_num, replace=False)
        elif len(candidate_dict_celltype["celltype"]) > 0:
            select_cell = np.random.choice(candidate_dict_celltype["images"], real_select_num, replace=True)
        else:
            if len(candidate_dict_celltype_bk["celltype"]) > real_select_num:
                select_cell = np.random.choice(candidate_dict_celltype_bk["images"], real_select_num, replace=False)
            elif len(candidate_dict_celltype_bk["celltype"]) > 0:
                select_cell = np.random.choice(candidate_dict_celltype_bk["images"], real_select_num, replace=True)
        
        if len(candidate_dict_risk["risk"]) > real_select_num:
            select_risk = np.random.choice(candidate_dict_risk["images"], real_select_num, replace=False)
        elif len(candidate_dict_risk["risk"]) > 0:
            select_risk = np.random.choice(candidate_dict_risk["images"] + candidate_dict_celltype_bk["images"], real_select_num, replace=True)
        else:
            select_risk = np.random.choice(candidate_dict_celltype_bk["images"] + candidate_dict_risk["images"], real_select_num, replace=True)
        
        selected_im = np.concatenate((select_cell, select_risk))

        return selected_im

    def __getitem__(self, index):
        subID = self.subject_list[index]
        sublabel = self.subject_label_list[index]
        try:
            sublabel = self.label_dict[sublabel]  # parse label
        except Exception as e:
            pass

        replace_prefix = "./data/pathology_2021/"
        replace_with = "./data//Pathology/pathology_2021/"

        feature_folder = self.feature_path_list[index].replace(replace_prefix, replace_with)
        celltype_csv = Path(feature_folder) / str(self.selective_features[0]) / "prd.csv"
        if self.target == "glioma":
            risk_csv = Path(feature_folder) / str(self.selective_features[1]) / "prd.csv"
            select_image_list = self._select_image_case(celltype_csv=celltype_csv, risk_csv=risk_csv, select_num=self.select_number)
        else:
            select_image_list = self._select_image_case_celltype(celltype_csv=celltype_csv, select_num=self.select_number)

        #load image as bags
        load_im_list = None
        for image_path in select_image_list:
            try:
            
                im = Image.open(image_path)
                im = augmentation(im)  # augmentaation
                im = self.transform(im)  # .permute(1, 2, 0)  # 处理图像
                im = im.unsqueeze(0)
                if load_im_list is None:
                    load_im_list = im
                else:
                    load_im_list = torch.cat((load_im_list, im), dim=0)
                    
            except Exception as e:
                print(e, image_path)

        im_bag = load_im_list
        return subID, im_bag, sublabel
