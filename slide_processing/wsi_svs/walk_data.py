import os
import pandas as pd

if __name__ == "__main__":
    subforders = ["GBM"]
    path = "/data/tianyang/patho_20230504/"
    sub_slides = {"slides": []}
    for root, dir, file in os.walk(path):
        for f in file:
            if "Default_0.tif" in f:
                sub_slides["slides"].append(os.path.join(root, f))
                print(os.path.join(root, f))

    pd.DataFrame(sub_slides).to_csv("/data/tianyang/patho_20230504/slides_all_20230609.csv")