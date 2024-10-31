import torch
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from patch_level.models.patch_classifier import PatchClassifier
import os
import glob
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

class Visualizer:
    def __init__(self):
        pass

    def visualize(self, result_df, output_path=None):
        # show the classification view
        if isinstance(result_df, str):  # input path
            df = pd.read_csv(result_df)
        else:
            df = result_df
        
        file_names = df['images'].tolist()
        row_col = np.array([self._name2vec(i) for i in file_names])

        prds = df['prd'].tolist()

        vis = np.zeros(np.max(row_col, axis=0) + 1)
        
        for ind, p in enumerate(prds):
            if p > 0:
                # print(row_col[ind][0], row_col[ind][1], p)
                vis[row_col[ind][0], row_col[ind][1]] = p
            else:
                vis[row_col[ind][0], row_col[ind][1]] = 10087

        masking = np.zeros_like(vis)  # rgba a channel
        masking[vis > 0] = 255

        vis = self._int2binary(vis)
        cv2.normalize(vis, vis, 0, 255, cv2.NORM_MINMAX)
        vis = cv2.merge([vis[:, :, 0].astype(np.uint8), vis[:, :, 1].astype(np.uint8),
                            vis[:, :, 2].astype(np.uint8), masking.astype(np.uint8)])

        if output_path is not None:
            cv2.imwrite(output_path, vis)

        return vis

    def _int2binary(self, mask_slice):
        color_dict = {
            '1': [0, 1, 1],  # yellow
            '2': [0, 0, 1],  # red
            '3': [0.79, 0.75, 1], # pink
            '4': [0, 1, 0],  # green
            '5': [1, 0, 1],  # meihong
            '6': [1, 1, 0],  #
            '0': [1, 1, 1],
            # '0': [0.3, 0.7, 0]
        }
        mask_slice_rgb = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
        unique_label = np.unique(mask_slice)
        for l in unique_label:
            if l > 0:
                mask_slice_rgb[mask_slice == l] = color_dict[str(int(l % 7))]

        return mask_slice_rgb

    def _name2vec(self, file_name):
        file_name = file_name.split('/')[-1]
        name_noext = file_name.replace('.JPG', '')
        R_C = name_noext.split('_')
        row_ind = int(R_C[0].replace('R', ''))
        col_ind = int(R_C[1].replace('C', ''))
        return [col_ind, row_ind]
        # return [row_ind, col_ind]


def apply_patch_level_single_case(im_list, model):
    """ list of png files
    """
    # test_transform = transforms.Compose([
    #         transforms.Resize((912, 1360)),
    #         transforms.CenterCrop((896, 1344)),
    #         # transforms.Resize((456, 675)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    test_transform = transforms.Compose([
            transforms.Resize((1216, 1812)),
            transforms.CenterCrop((704, 704)),
            # transforms.Resize((456, 675)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    case_result = {"images": [], "prd": [], "prob":[]}
    for im in tqdm(im_list):
        # transform image
        im_id = os.path.basename(im).replace(".JPG", "")
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = Image.open(im)
        im = test_transform(im).unsqueeze(0).cuda()

        output = model.forward(im)
        output = torch.softmax(output, dim=1)  # .data.cpu().numpy()[:, 1]=

        prob = torch.max(output, 1)[0].data.cpu().numpy().tolist()
        pred = torch.max(output, 1)[1].data.cpu().numpy().tolist()
        
        case_result["images"].extend([im_id])
        case_result["prd"].extend(pred)
        case_result["prob"].extend(prob)
    
    df = pd.DataFrame(case_result)
    return df

def apply_model():
    # model_info = {"model_path": "/data/tianyang/patho_cryo/patch_tnt/checkpoints/step_11700.chkpt",
    #               "model_type": "tumour", 
    #               "save_name": "Cryotumour", 
    #               "num_classes": 3
    #               }

    model_info = {"model_path": "/data/tianyang/patho_cryo/patch_risk3grade/checkpoints/epoch_50.chkpt",
                  "model_type": "risk", 
                  "save_name": "Cryorisk_grade3",
                  "num_classes": 5
                  }
    
    model = PatchClassifier(num_classes=model_info["num_classes"], pretrained=False).eval().cuda()
    model_chkpt = torch.load(model_info["model_path"])
    model.load_state_dict(model_chkpt["state_dict"])
    
    df = pd.read_csv("/data/tianyang/patho_db/retrospective_glioma_matched.csv")
    for id, row in df.iterrows():
        if id >= 0:
            subID = row["SubID"]
            pref_satisfy = True
            for prefix in ["N10", "N11", "N12", "N13"]:
                if prefix in subID:
                    pref_satisfy = False
                    break
            try:
                if pref_satisfy:
                    patch_folder = row["sub_folder"]
                    print("{}/{}".format(id, len(df)), subID, patch_folder)

                    output_path = os.path.join(patch_folder, "features", model_info["save_name"])

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                        im_list = sorted(glob.glob(os.path.join(patch_folder, "*.JPG")))
                        result_df = apply_patch_level_single_case(im_list, model)
                        result_df.to_csv(os.path.join(output_path, "prd.csv"))

                        visualizer = Visualizer()
                        visualizer.visualize(result_df, os.path.join(output_path, "prd_{}.png".format(model_info["save_name"])))

            except Exception as e:
                print(e)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # im_list = sorted(glob.glob(os.path.join(patch_dir, "*.JPG")))
    # result_df = apply_patch_level_single_case(im_list, model)
    # result_df.to_csv(os.path.join(output_path, "prd.csv"))

    # visualizer = Visualizer()
    # visualizer.visualize(result_df, os.path.join(output_path, "prd_{}.png".format(model_info["save_name"])))

def apply_model_exclude_bg():
    # model_info = {"model_path": "/data/tianyang/patho_cryo/patch_tnt/checkpoints/step_11700.chkpt",
    #               "model_type": "tumour", 
    #               "save_name": "Cryotumour", 
    #               "num_classes": 3
    #               }

    # model_info = {"model_path": "/data/tianyang/patho_cryo/patch_risk3grade_0202/checkpoints/epoch_35.chkpt",
    #               "model_type": "risk", 
    #               "save_name": "Cryorisk_grade3_0202",
    #               "num_classes": 5,
    #               "bg_fiter_name": "bg.csv"
    #               }
    # celtype 0204 huashan 
    # model_info = {
    #     "model_path": "/data/tianyang/patho_cryo/patch_celltype_0202/checkpoints/epoch_55.chkpt",
    #     "model_type": "celltype", 
    #     "save_name": "Cryo_celltype_0204", 
    #     "num_classes": 4, 
    #     "bg_fiter_name": "bg.csv"
    # }
    # TCGA round1 label;
    # model_info = {
    #         "model_path": "/data/tianyang/patho_cryo/patch_huashan_tcga_celltype/checkpoints/epoch_280.chkpt",
    #         "model_type": "celltype",
    #         "num_classes": 4,
    #         "save_name": "Cryo_huashan_tcga_celltype", 
    #         "bg_fiter_name": "bg.csv"
        # }
    # HUASHAN
    model_info = {
        "model_path": "/data/tianyang/patho_cryo_prev/patch_huashan_tcga_convnext_celltype_0428/checkpoints/epoch_213.chkpt",
        "model_type": "celltype",
        "num_classes": 4,
        "save_name": "celltype_0514", 
        "bg_fiter_name": "bg.csv",
        "model_name": "convnext_tiny", 
        "ce_loss_weight": torch.tensor([0.25, 0.25, 0.25, 0.25])
    }

    # model_info = {
    #     "model_path": "/data/tianyang/patho_cryo/patch_huashan_tcga_3grade_0430/checkpoints/epoch_900.chkpt",
    #     "model_type": "risk",
    #     "num_classes": 5,
    #     "save_name": "risk_0506", 
    #     "bg_fiter_name": "bg.csv",
    #     "model_name": "convnext", 
    #     "ce_loss_weight": torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3])
    # }


    
    model = PatchClassifier(num_classes=model_info["num_classes"], model_name=model_info["model_name"], pretrained=None, ce_loss_weight=model_info["ce_loss_weight"]).eval().cuda()
    model_chkpt = torch.load(model_info["model_path"])
    model.load_state_dict(model_chkpt["state_dict"])
    
    # df = pd.read_csv("/mnt/data198/tianyang/patho_tcga/tcga_slide_info.csv")
    # df = pd.read_csv("/data/tianyang/patho_cryo/clear_label_20240424/huashan_validation_slide.csv")
    # df = pd.read_csv("/data/tianyang/patho_cryo_prev/clear_label_20240424/huashan_retrospective_slide.csv")
    df = pd.read_csv("/data/tianyang/patho_cryo/slide_label_202406/Huashan_prospective_slide_retile.csv")
    for id, row in df.iterrows():
        if id >= 0:
            # subID = row["case_submitter_id"]
            subID = row["SubID"]
            pref_satisfy = False
            try:
                # patch_folder = row["CryoSlidePath"]
                patch_folder = row["sub_folder"]
                if "redo" in patch_folder:
                    print("{}/{}".format(id, len(df)), subID, patch_folder)

                    output_path = os.path.join(patch_folder, "features", model_info["save_name"])
                    print(os.path.join(patch_folder, model_info["bg_fiter_name"]))

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    if os.path.exists(os.path.join(patch_folder, model_info["bg_fiter_name"])):
                        print("filter bg")
                        df_bg = pd.read_csv(os.path.join(patch_folder, model_info["bg_fiter_name"]))
                        df_nonbg = df_bg[df_bg.prd != 0]
                        im_list = df_nonbg["images"].to_list()
                        im_list = [os.path.join(patch_folder, "{}.JPG".format(xx)) for xx in im_list]
                    else:
                        print("no bg image found, apply all patch forward")
                        im_list = sorted(glob.glob(os.path.join(patch_folder, "*.JPG")))

                    result_df = apply_patch_level_single_case(im_list, model)
                    result_df.to_csv(os.path.join(output_path, "prd.csv"))

                    visualizer = Visualizer()
                    visualizer.visualize(result_df, os.path.join(output_path, "prd_{}.png".format(model_info["save_name"])))

            except Exception as e:
                print(e)


if __name__ == "__main__":
    apply_model_exclude_bg()