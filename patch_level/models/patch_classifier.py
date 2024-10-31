# from torchvision.models.convnext import convnext_base, convnext_tiny
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from patch_level.models.Convnext import convnext_tiny, convnext_base, convnext_small
from patch_level.models.vision_transformer import vit_small, LinearClassifier
from patch_level.models.vision_transformer import vit_tiny
from pretrain_dino.train_dino import *
import pytorch_lightning as pl
import torch
import torch.nn as nn
from patch_level.loss.ce_circle import CircleLoss, CeCircle
# from patch_level.loss.circle_loss import CircleLoss, convert_label_to_similarity


class PatchClassifier(pl.LightningModule):
    def __init__(self, num_classes, model_name="convnext", pretrained=True, ce_loss_weight=None, learning_rate=5e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.ce_loss_weight = ce_loss_weight
        # self.loss_func = torch.nn.CrossEntropyLoss(weight=weight)
        self.loss_func = CeCircle(ce_weight=self.ce_loss_weight)
        
        if model_name == "convnext_tiny":
            if pretrained is not None:
                pretrained = True  # use imagenet default parameters
            self.model = convnext_tiny(pretrained=pretrained)
            self.model.redefine_classifier(num_classes=num_classes)
        elif model_name == "convnext_small":
            if pretrained is not None:
                pretrained = True  # use imagenet default parameters
            self.model = convnext_small(pretrained=pretrained)
            self.model.redefine_classifier(num_classes=num_classes)

        elif model_name == "convnext_base":
            if pretrained is not None:
                pretrained = True  # use imagenet default parameters
            self.model = convnext_base(pretrained=pretrained)
            self.model.redefine_classifier(num_classes=num_classes)
    
            
        elif model_name == "vit_with_linear":
            self.model = vit_tiny(patch_size=16)  # vit small embed 384 dimension, vit tiny embed 192 dimension
            self.classifier = LinearClassifier(dim=192, num_labels=num_classes)
            if pretrained is not None:
                if not isinstance(pretrained, str):
                    print("Input path to checkpoint for vit pretrained")
                    raise ValueError
                self._load_pretrained_vit(pretrained_path=pretrained)
        else:
            raise NotImplementedError
    
    def _load_pretrained_vit(self, pretrained_path):
        state_dict = torch.load(pretrained_path)

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        ## must have for training
        lr = self.learning_rate
        if self.model_name in ["convnext_base", "convnext_small", "convnext_tiny"]:
            # optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
            return optimizer


        elif self.model_name == "vit_with_linear":
            param_dict = [
                {"params": self.model.parameters(), "lr": lr * 0.8}, 
                {"params": self.classifier.parameters(), "lr": lr}, 
            ]
            optimizer = torch.optim.Adam(param_dict)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        
    def training_step(self, batch) -> STEP_OUTPUT:
        im = batch[0].cuda()
        label = batch[1].long().cuda()

        feature, output = self.model(im)
        feature = feature.squeeze()
        feature = torch.nn.functional.normalize(feature)

        if self.model_name == "vit_with_linear":
            output = self.classifier(output)

        weight = self.ce_loss_weight.cuda()
        ce = torch.nn.CrossEntropyLoss(weight=weight)
        ce_loss = ce(output, label)
        loss = self.loss_func(output, feature, label)
        circle_l = CircleLoss()
        circle_loss = circle_l(feature, label) 
        
        self.log("patch_loss_Circle", circle_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("patch_loss_CE", ce_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        loss = ce_loss  + 0.0025 * circle_loss
        return loss

    # def validation_step(self, *args: Any, **kwargs: Any):

    def forward(self, x) -> Any:
        if self.model_name in ["convnext_base", "convnext_small", "convnext_tiny"]:
            feature, output = self.model(x)
            return output
        elif self.model_name == "vit_with_linear":
            # forward vit
            features = self.model(x)
            output = self.classifier(features)
            return output


if __name__ == "__main__":
    a = torch.randn((1, 3, 1536, 2274))
    model = convnext_tiny(pretrained=True)
    model.redefine_classifier(num_classes=3)

    out= model(a)

    from IPython import embed
    embed()