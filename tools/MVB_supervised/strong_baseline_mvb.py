#!/usr/bin/env python
# encoding: utf-8
"""
Strong baseline training for SuitcaseReID dataset
Modified from strong_baseline/main.py
"""
import argparse
import shutil
import sys

sys.path.append("/data1/zhaofanghan/SuitcaseReID/OpenUnReID")

import time
from datetime import timedelta
from pathlib import Path
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb  # Import Weights & Biases

from openunreid.apis import BaseRunner, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize, get_dist_info
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger


def parse_config():
    parser = argparse.ArgumentParser(description="SuitcaseReID supervised training")
    parser.add_argument("--config", help="train config file path", 
                        default=osp.join(osp.dirname(__file__), 'config.yaml'))
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models"
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    
    # Use a specific work directory for SuitcaseReID
    if not args.work_dir:
        # 获取当前时间
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 设置工作目录
        args.work_dir = "suitcase_supervised_mvb_" + current_time
    
    if cfg.LOGS_ROOT is not None:
        cfg.work_dir = Path(cfg.LOGS_ROOT) / args.work_dir
    else:
        cfg.work_dir = Path("../logs") / args.work_dir

    mkdir_if_missing(cfg.work_dir)
    
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # Copy config file for reference
    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


class MVBBaseRunner(BaseRunner):
    """
    Extended BaseRunner with W&B logging capabilities
    """
    
    def train_step(self, iter, batch):
        """
        Override the train_step method to add W&B logging
        """
        # Call the original train_step method
        (total_loss, meters) = super().train_step(iter, batch)
        
        # Log metrics to W&B
        rank, _, _ = get_dist_info()
        if rank == 0 and iter % self.print_freq == 0:
            # Log loss metrics
            for k, v in meters.items():
                wandb.log({f"train/{k}": v})
            wandb.log({"train/loss": total_loss})
            
            # Log learning rate
            wandb.log({f"train/lr": self.optimizer.param_groups[0]['lr']})

        
        return (total_loss, meters)
    
    def val(self):
        """
        Override the validation method to add W&B logging
        """
        # Call the original val method
        results = super().val()
        
        # Log validation metrics to W&B if available
        rank, _, _ = get_dist_info()
        if rank == 0 and results is not None:
            for (mAP, cmc) in [results]:
                wandb.log({
                    f"val/mAP": mAP,
                    f"val/rank1": cmc[0],
                    f"val/rank5": cmc[4],
                    f"val/rank10": cmc[9]
                })
        
        return results


def main():

    # CUDA_VISIBLE_DEVICE=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /data1/zhaofanghan/OpenUnReID/tools/SuitcaseReID_supervised/strong_baseline_suitcase.py --config /data1/zhaofanghan/OpenUnReID/tools/SuitcaseReID_supervised/suitcase_config.yaml --launcher pytorch
    
    # torchrun version: 
    # Use torchrun instead of torch.distributed.launch, becuase it sets the local rank in environemt variable
    # not automatically inject --local_rank
    
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    # --nproc_per_node=4 \
    # /data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised/strong_baseline_mvb.py \
    # --config /data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised/mvb_config.yaml \
    # --launcher pytorch \
    # --resume-from /data1/zhaofanghan/SuitcaseReID/OpenUnReID/logs/suitcase_supervised_mvb_2025-04-22_23-15-04/model_best.pth
    torch.autograd.set_detect_anomaly(True)
    start_time = time.monotonic()

    # Init configuration
    args, cfg = parse_config()
    
    # Init distributed training
    dist = init_dist(cfg)
    rank, world_size, is_dist = get_dist_info()
    
    # Set random seed for reproducibility
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # Init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)
    
    # Initialize W&B - only on the main process in distributed training
    if rank == 0:
        wandb.init(
            project="Suitcase ReID",
            name="MVB_" + str(cfg.work_dir).split("/")[-1],
            config=cfg,
        )
        print("Weights & Biases initialized for project: Suitcase ReID")
    
    # Setup TensorBoard for visualization
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=osp.join(cfg.work_dir, 'tensorboard'))
        print(f"TensorBoard logs will be saved to {osp.join(cfg.work_dir, 'tensorboard')}")

    # Build train loader - using the supervised SuitcaseReID dataset
    print("Building dataloaders for SuitcaseReID training...")
    train_loader, train_sets = build_train_dataloader(cfg)
    print(f"Train loader built with {len(train_sets)} dataset(s)")

    # Count number of classes (suitcase IDs)
    num_classes = 0
    for idx, dataset in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # For unsupervised datasets - should not be used in this supervised setting
            num_classes += len(dataset)
            print(f"Warning: Using unsupervised dataset with {len(dataset)} samples")
        else:
            # For supervised dataset - use actual number of suitcase IDs
            num_classes += dataset.num_pids
            print(f"Using supervised dataset with {dataset.num_pids} suitcase IDs")
    
    print(f"Training model with {num_classes} classes")

    # Build model - initialize with ImageNet weights if specified
    # ResNet50 with GeM pooling works well for suitcases
    model = build_model(cfg, num_classes, init=cfg.MODEL.source_pretrained)
    model.cuda()
    
    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    # Build optimizer - Adam works well for suitcase ReID
    optimizer = build_optimizer([model], **cfg.TRAIN.OPTIM)

    # Build learning rate scheduler - Cosine works well
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # Build loss functions - CE + Triplet is effective for suitcase ReID
    criterions = build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True)

    # Build runner
    runner = MVBBaseRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        reset_optim=True,
    )

    # Watch model with W&B to track parameters and gradients
    if rank == 0:
        wandb.watch(model, log="all", log_freq=100)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        runner.resume(args.resume_from)

    # Start training
    print("Starting SuitcaseReID training...")
    runner.run()

    # Load the best model
    best_model_path = cfg.work_dir / "model_best.pth"
    print(f"Checking for best model at {best_model_path}")
    
    if osp.isfile(best_model_path):
        print(f"Loading best model from {best_model_path}")
        runner.resume(best_model_path)
    else:
        print(f"Warning: Best model file not found at {best_model_path}. Using current model state for evaluation.")

    # Final testing
    print("Evaluating on SuitcaseReID test set...")
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        print(f"Testing on dataset: {cfg.TEST.datasets[i]}")
        print(f"Query set: {len(query)} images, Gallery set: {len(gallery)} images")
        cmc, mAP = test_reid(
            cfg, model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i]
        )
        
        # Log results to TensorBoard and W&B
        if rank == 0:
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar(f'Test/{cfg.TEST.datasets[i]}/mAP', mAP, 0)
                for r in [1, 5, 10]:
                    if r < len(cmc):
                        writer.add_scalar(f'Test/{cfg.TEST.datasets[i]}/Rank-{r}', cmc[r-1], 0)
            
            # Log to W&B
            wandb.log({
                f'test/{cfg.TEST.datasets[i]}/mAP': mAP,
                f'test/{cfg.TEST.datasets[i]}/rank1': cmc[0],
                f'test/{cfg.TEST.datasets[i]}/rank5': cmc[4] if len(cmc) > 4 else None,
                f'test/{cfg.TEST.datasets[i]}/rank10': cmc[9] if len(cmc) > 9 else None,
            })
            
            # Save model artifact in W&B
            if i == 0:  # Only save once
                wandb.save(str(best_model_path))

    # Print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
    # cd /data1/zhaofanghan/OpenUnReID/tools/SuitcaseReID_supervised && python strong_baseline_suitcase.py --config suitcase_config.yaml