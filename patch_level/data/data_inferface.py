## the aim of this file is to create an interface for pytorch lightening data module and python dataset
import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=16,
                 dataset='',
                 val_imlist="",
                 **kwargs):
        super().__init__()
        
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs
        self.val_imlist = val_imlist
        self.kwargs.pop("batch_size")      

        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.trainset = self.data_module_train
        # self.valset = self.data_module_val

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            if self.dataset == "patch_tumour":
                from patch_level.data.patch_dataset import PatchDataset
                self.data_module_train = PatchDataset(imlist_file=self.kwargs["train_file"], is_train=True, target=self.kwargs["target"])  # 
                # self.data_module_val = PatchDataset(imlist_file=self.kwargs["val_file"], is_train=False, target=self.kwargs["target"])

            else:
                raise NotImplementedError
            # self.data_module = getattr(importlib.import_module(
            #     '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    # def instancialize(self, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.kwargs.
    #     """
    #     class_args = inspect.getargspec(self.data_module.__init__).args[1:]
    #     inkeys = self.kwargs.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = self.kwargs[arg]
    #     args1.update(other_args)
    #     return self.data_module(**args1)
    
# def define_synthmr_datamodule(datasetname, cfg):
if __name__ == "__main__":
    
    datamodule = DInterface(dataset='patch_tumour', 
                            batch_size=6, 
                            imlist_file="/mnt/data214/tianyang/patho_cryo/patch_huashan_tcga_celltype/huashan_tcga_patch_train_r3.csv",  
                            target="tumour")
    datamodule.setup()
    from IPython import embed
    embed()