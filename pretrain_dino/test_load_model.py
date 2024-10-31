from train_dino import *
import torch

if __name__ == "__main__":
    chkpt = torch.load("/mnt/data70/tianyang/patho_cryo_202401/pretrain_DINO/checkpoints/checkpoint0020.pth")
    from IPython import embed
    embed()