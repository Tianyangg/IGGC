## 通过统计prediction中celltype的比重评估测试数据的celltype效果

import pandas as pd
from pathlib import Path

def evaluate_celltype():
    df = pd.read_csv("/data/tianyang/patho_cryo/slide_label_202406/Huashan_retrospective_cohort_consistent_relabel_trainval.csv")
    df_val = df[df.patch_trained == 0]
    celltype_folder = "celltype_0514"
    df_val["proportion_o"] = 0
    for id, row in df_val.iterrows():
        feature_folder = Path(row["sub_folder"])
        prd_csv = feature_folder / "features" / celltype_folder / "prd.csv"
        df_prd = pd.read_csv(prd_csv)
        prd_labels_O = len(df_prd[df_prd.prd == 2])
        prd_labels_A = len(df_prd[df_prd.prd == 3])

        total = prd_labels_O + prd_labels_A
        proportion_o = prd_labels_O / (total + 1e-5)
        df_val.loc[id, "proportion_o"] = proportion_o
    
        
    df_val.to_csv("/data/tianyang/patho_cryo/slide_label_202406/Huashan_retrospective_cohort_consistent_relabel_val_celltype_0514.csv")
        
if __name__ == "__main__":
    evaluate_celltype()