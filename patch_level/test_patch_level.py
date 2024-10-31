import torch
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from patch_level.models.patch_classifier import PatchClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(gt, prd, num_class):
    print(classification_report(gt, prd))
    mat = confusion_matrix(gt, prd)

    sns.heatmap(mat, annot=True, fmt='d', cmap="Blues")
    plt.ylabel("GroundTruth")
    plt.xlabel("Prediction")
    ticks_ind = [x + 0.5 for x in range(len(mat))]

    # plt.yticks(ticks_ind, list(range(num_class)))
    # plt.xticks(ticks_ind, list(range(num_class)))

    plt.show()  


def apply_patch_level_single_case(im_list, model, gt_list):
    """ list of png files
    """
    test_transform = transforms.Compose([
            transforms.Resize((912, 1360)),
            transforms.CenterCrop((896, 1344)),

            # transforms.Resize((456, 675)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    case_result = {"im_path": [], "prd": [], "prob":[]}
    if gt_list is not None:
        case_result.update({"gt": []})
    for id, im in tqdm(enumerate(im_list)):
        # transform image
        im_id = os.path.basename(im).replace(".JPG", "")
        im_p = im
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = Image.open(im)
        im = test_transform(im).unsqueeze(0).cuda()

        output = model.forward(im)
        output = torch.softmax(output, dim=1)  # .data.cpu().numpy()[:, 1]=

        prob = torch.max(output, 1)[0].data.cpu().numpy().tolist()
        pred = torch.max(output, 1)[1].data.cpu().numpy().tolist()
        
        case_result["im_path"].extend([im_p])
        case_result["prd"].extend(pred)
        case_result["prob"].extend(prob)
        if gt_list is not None:
            # print(pred, gt_list[id])
            case_result["gt"].extend([gt_list[id]])
    
    df = pd.DataFrame(case_result)
    return df

def apply_model():
    # model_info = {"model_path": "/data/tianyang/patho_cryo/patch_tnt/checkpoints/step_11700.chkpt",
    #               "model_type": "tumour", 
    #               "save_name": "Cryotumour", 
    #               "num_classes": 3
    #               
    steps = [100, 200, 300, 500]
    for step in steps:
        # model_info = {
        #     "model_path": "/data/tianyang/patho_cryo/patch_risk3grade_0202/checkpoints/epoch_{}.chkpt".format(step),
        #     "model_type": "risk_grade3",
        #     "num_classes": 5,
        #     "gt_list_name": "patch_label_risk"
        # }
        print("Validating: ", step)
        model_info = {
            "model_path": "/data/tianyang/patho_cryo/patch_tcga_celltype//checkpoints/epoch_{}.chkpt".format(step),
            "model_type": "celltype",
            "num_classes": 4,
            "gt_list_name": "path_label_celltype"
        }

        eval_csv = "/data/tianyang/patho_cryo/patch_risk3grade/patch_anno_retro_validation_labeled_0122.csv"
        
        model = PatchClassifier(num_classes=model_info["num_classes"], pretrained=False).eval().cuda()
        model_chkpt = torch.load(model_info["model_path"])
        model.load_state_dict(model_chkpt["state_dict"])
        
        val_df = pd.read_csv(eval_csv)
        im_list = val_df["im_path"]
        gt_list = val_df[model_info["gt_list_name"]]
        # replace filename
        im_list = [x.replace("/data/tianyang/", "/mnt/data198/tianyang/") for x in im_list]

        res = apply_patch_level_single_case(im_list, model=model, gt_list=gt_list)
        res.to_csv("/data/tianyang/patho_cryo/patch_celltype_0202/val_path_res_{}_{}.csv".format(model_info["model_type"], step), index=False)
        print(classification_report(res["gt"], res["prd"]))
    # im_list = sorted(glob.glob(os.path.join(patch_dir, "*.JPG")))
    # result_df = apply_patch_level_single_case(im_list, model)
    # result_df.to_csv(os.path.join(output_path, "prd.csv"))

if __name__ == "__main__":
    apply_model()
    # df = pd.read_csv("/data/tianyang/patho_cryo/patch_celltype_0202/val_path_res_celltype_10.csv")
    # gt = df["gt"]
    # prd= df["prd"]
    # plot_confusion_matrix(gt, prd, num_class=4)
