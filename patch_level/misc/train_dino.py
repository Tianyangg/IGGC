import torch
import torch.nn as nn
# from dataset import PatchDataset
from torch.utils.data import DataLoader
# from utils import MultiCropWrapper, has_batchnorms
# from vision_transformer import vit_tiny, DINOHead
# from loss import DINOLoss
import utils
import time
import os
import datetime
import json
from pathlib import Path
import sys
import math


# PUT ALL Hyperparmeters here temporary
class Args:
    def __init__(self):
        self.batch_size = 128
        self.data_csv_path = "/data/tianyang/patho_cryo_202401/pretrain_DINO/pretrain_patch_tcga_retrospective.csv"
        self.drop_path_rate = 0.1
        self.projection_head_out_dim = 65536
        self.local_crops_number = 8
        self.warmup_teacher_temp = 0.04
        self.teacher_temp = 0.07
        self.warmup_teacher_temp_epochs = 30
        self.epochs = 400
        self.norm_last_layer = False

        self.lr = 0.0005
        self.batch_size_per_gpu = 32
        self.min_lr = 1e-6
        self.warmup_epochs = 10
        self.weight_decay = 0.04
        self.weight_decay_end = 0.4

        self.momentum_teacher = 0.996

        self.use_fp16 = False

        self.output_dir = "/data/tianyang/patho_cryo_202401/pretrain_DINO/checkpoints2/"
        self.saveckp_freq = 10

        self.dist_url = "env://"

        self.clip_grad = 3.0
        self.freeze_last_layer = 1
        

def train_dino(args):
    utils.init_distributed_mode(args)
    # define dataset
    dataset = PatchDataset(args.data_csv_path)
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=False,
        num_workers=64
    )

    print(f"Data loaded: there are {len(dataset)} images.")

    # define network
    student = vit_tiny(patch_size=16, drop_path_rate=args.drop_path_rate)
    teacher = vit_tiny(patch_size=16)
    embed_dim = student.embed_dim

    student = MultiCropWrapper(
        student, 
        DINOHead(
        embed_dim,
        args.projection_head_out_dim,
        use_bn=False,
        norm_last_layer=args.norm_last_layer,
    ))

    teacher = MultiCropWrapper(
        teacher, 
        DINOHead(embed_dim, args.projection_head_out_dim, use_bn=False))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    student = nn.parallel.DataParallel(student)
    teacher = nn.parallel.DataParallel(teacher)
    teacher_without_ddp = teacher.module

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # Define Loss
    dino_loss = DINOLoss(
        args.projection_head_out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # define optimizer
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    # define scheduler
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(train_loader))
    print(f"Loss, optimizer and schedulers ready.")

    start_epoch = 0
    start_time = time.time()
    print("Starting DINO training !")
    fp16_scaler = None

    for epoch in range(start_epoch, args.epochs):
        # train_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    args = Args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)