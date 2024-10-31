from patch_level.data.data_inferface import DInterface
from patch_level.models.patch_classifier import PatchClassifier
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
import os
import torch
from pytorch_lightning.callbacks import LearningRateMonitor


class MySavingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        # pl_module.configure_optimizers()
    
    def on_train_epoch_end(self, trainer, pl_module):
        print("Training epoch end")
        print("Global step", pl_module.global_step)
        current_epoch = pl_module.current_epoch
        checkpoint_folder = os.path.join(trainer.default_root_dir, "checkpoints")
        trainer.save_checkpoint(os.path.join(checkpoint_folder, "epoch_{}.chkpt".format(current_epoch)))
        
        
        # if pl_module.global_step % 2200 == 0:
        # # trainer.save_checkpoint("/data/tianyang/MRSynth/latent_diffusion/vae/latent_vae2/{}.chkpt".format(pl_module.global_step))
        #     checkpoint_folder = os.path.join(trainer.default_root_dir, "checkpoints")
        #     if not os.path.exists(checkpoint_folder):
        #         try:
        #             os.makedirs(checkpoint_folder)
        #         except Exception as e:
        #             print(e)
                    
        #     trainer.save_checkpoint(os.path.join(checkpoint_folder, "step_{}.chkpt".format(pl_module.global_step)))


def train():
    # train_info = {
    #     "save_dir": "/data/tianyang/patho_cryo/patch_risk3grade/", 
    #     "target": "risk_grade3_new",
    #     "train_csv": "/home/tianyang/Code/patho_cryo/patch_anno_retro_0120.csv",
    #     "val_csv": "",
    #     # "val_csv": "/data/tianyang/patho_cryo/patch_risk3grade/patch_anno_retro_validation_labeled_0122.csv",
    #     "num_classes": 5
    # }
    ## risk 0510
    # train_info = {
    #     "save_dir": "/mnt/data214/tianyang/patho_cryo/patch_huashan_tcga_3grade_0522/", 
    #     "target": "risk_grade3_new",
    #     "train_csv": "/mnt/data214/tianyang/patho_cryo/patch_level_label_0430/patch_risk_0510.csv",
    #     "val_csv": "",
    #     "model_name": "convnext_tiny", 
    #     "model_pretrained":  "/mnt/data214/tianyang/patho_cryo/pretrain_vit/checkpoint0020_student.pth", 
    #     # "val_csv": "/data/tianyang/patho_cryo/patch_risk3grade/patch_anno_retro_validation_labeled_0122.csv",
    #     "num_classes": 5, 
    #     "ce_loss_weight": torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3]), 
    #     "learning_rate": 5e-4, 
    # }
    ### celltype 0510
    # train_info = {
    #     "save_dir": "/mnt/data214/tianyang/patho_cryo/patch_huashan_tcga_convnext_celltype_0428_continue/", 
    #     "target": "celltype",
    #     "train_csv": "/mnt/data214/tianyang/patho_cryo/patch_level_label_0430/patch_celltype_0510.csv",
    #     "val_csv": "",
    #     "model_name": "convnext_tiny", 
    #     "model_pretrained":  "/mnt/data214/tianyang/patho_cryo/pretrain_vit/checkpoint0020_student.pth", 
    #     # "val_csv": "/data/tianyang/patho_cryo/patch_risk3grade/patch_anno_retro_validation_labeled_0122.csv",
    #     "num_classes": 4, 
    #     "ce_loss_weight": torch.tensor([0.25, 0.25, 0.25, 0.25]), 
    #     "learning_rate": 1e-4, 
    #     "checkpoint_resume": "/mnt/data214/tianyang/patho_cryo/patch_huashan_tcga_convnext_celltype_0428/checkpoints/epoch_235.chkpt"
    # }
    
    # celltype convnext small
    # train_info = {
    #     "save_dir": "/data/tianyang/patho_cryo/patch_huashan_tcga_convnextsmall_celltype_0607/", 
    #     "target": "celltype",
    #     "train_csv": "/data/tianyang/patho_cryo/patch_level_label_0430/patch_celltype_0607_3.csv",
    #     "val_csv": "",
    #     "model_name": "convnext_small", 
    #     "model_pretrained":  "", 
    #     # "val_csv": "/data/tianyang/patho_cryo/patch_risk3grade/patch_anno_retro_validation_labeled_0122.csv",
    #     "num_classes": 4, 
    #     "ce_loss_weight": torch.tensor([0.25, 0.4, 0.2, 0.15]), # torch.tensor([0.25, 0.25, 0.25, 0.25]), 
    #     "learning_rate": 2e-4, 
    #     "checkpoint_resume": "/data/tianyang/patho_cryo/patch_huashan_tcga_convnextsmall_celltype_0607/checkpoints/epoch_94.chkpt"
    # }
    
    # identify gbm4
    train_info = {
        "save_dir": "/mnt/data214/tianyang/patho_cryo/patch_gbm_marker_0714/", 
        "target": "gbm_vs_other",
        "train_csv": "/mnt/data214/tianyang/patho_cryo/patch_level_label_0430/patch_gbm_0714_add6.csv",
        "val_csv": "",
        "model_name": "convnext_small", 
        "model_pretrained": "", 
        # "val_csv": "/data/tianyang/patho_cryo/patch_risk3grade/patch_anno_retro_validation_labeled_0122.csv",
        "num_classes": 2, 
        "ce_loss_weight": torch.tensor([0.5, 0.5]), 
        "learning_rate": 2e-4, 
    }

    # weight = torch.tensor([0.1, 0.3, 0.3, 0.3]).cuda()
    # weight = torch.tensor([0.1, 0.1, 0.15, 0.25, 0.4]).cuda()
    if not os.path.exists(train_info["save_dir"]):
        os.makedirs(train_info["save_dir"])

    save_dir = train_info["save_dir"]
    datamodule = DInterface(dataset='patch_tumour', 
                            batch_size=24, 
                            train_file=train_info["train_csv"],  
                            val_file=train_info["val_csv"],
                            target=train_info["target"])
    
    # model = PatchClassifier.load_from_checkpoint(
    #     num_classes=train_info["num_classes"], 
    #     ce_loss_weight=train_info["ce_loss_weight"], 
    #     model_name=train_info["model_name"], 
    #     pretrained=train_info["model_pretrained"], 
    #     learning_rate=train_info["learning_rate"],
    #     checkpoint_path=train_info["checkpoint_resume"]
    # )

    model = PatchClassifier(
        num_classes=train_info["num_classes"], 
        ce_loss_weight=train_info["ce_loss_weight"], 
        model_name=train_info["model_name"], 
        pretrained=train_info["model_pretrained"], 
        learning_rate=train_info["learning_rate"]
    )

    datamodule.setup()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(default_root_dir=save_dir, logger=True,  callbacks=[MySavingCallback(), lr_monitor],
                      devices="auto", accelerator="gpu", strategy="ddp_find_unused_parameters_true", max_epochs=100000, precision="16-mixed")
    trainer.logdir = save_dir
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()

