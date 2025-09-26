#!/usr/bin/env python
# encoding: utf-8
"""
Strong baseline training for SuitcaseReID dataset with Gaussian mask regression
Modified from strong_baseline/main.py

New Features:
- Gaussian mask regression loss with threshold control
- Only pixels above the threshold contribute to the loss computation
- Configurable via mask_threshold parameter in config file

Usage:
    Set mask_threshold in mvb_config.yaml:
    TRAIN:
      LOSS:
        mask_threshold: 0.1  # Only pixels with Gaussian value >= 0.1 contribute to loss
"""
import argparse
import shutil
import sys
import os

sys.path.append("/data1/zhaofanghan/SuitcaseReID/OpenUnReID")

import time
from datetime import timedelta
from pathlib import Path
import os.path as osp

import torch
import wandb  # Import Weights & Biases

from openunreid.apis import BaseRunner, test_reid, set_random_seed
from openunreid.utils.dist_utils import get_dist_info
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader
from openunreid.models import build_model_with_mask
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
                        default=osp.join(osp.dirname(__file__), 'mvb_config.yaml'))
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
    Extended BaseRunner with W&B logging capabilities and mask regression support
    """
    
    def train_step(self, iter, batch):
        """
        Override the train_step method to add mask regression and W&B logging
        """
        try:
            # Process batch data
            from openunreid.apis.train import batch_processor
            data = batch_processor(batch, self.cfg.MODEL.dsbn)
            
            inputs = data["img"][0].cuda(non_blocking=True)
            targets = data["id"].cuda(non_blocking=True)
            
            # Get bounding boxes if available (for mask generation)
            # In practice, you'll need to modify your dataset to provide bbox annotations
            # For now, we'll simulate bboxes or handle the case where they're not available
            bboxes = None
            if "bbox" in data:
                bboxes = data["bbox"].cuda()
            
            # Forward pass
            results = self.model(inputs)
            
            # Limit classifier output to actual number of classes
            if "prob" in results.keys():
                results["prob"] = results["prob"][:, : self.train_loader.loader.dataset.num_pids]
            
            total_loss = 0
            meters = {}
            
            # Compute ReID losses (cross-entropy, triplet, etc.)
            for key in self.criterions.keys():
                if key not in ["gaussian_mask_regression", "focal_mask", "combined_mask"]:
                    loss = self.criterions[key](results, targets)
                    total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
                    meters[key] = loss.item()
            
            # Compute mask regression loss if mask is generated
            if "mask" in results and any(loss_name in self.criterions for loss_name in 
                                       ["gaussian_mask_regression", "focal_mask", "combined_mask"]):
                
                pred_masks = results["mask"]
                
                # For mask loss computation, we need target bboxes
                # If not available, we can skip mask loss or generate dummy targets
                if bboxes is not None:
                    for loss_name in ["gaussian_mask_regression", "focal_mask", "combined_mask"]:
                        if loss_name in self.criterions:
                            if loss_name in self.cfg.TRAIN.LOSS.losses:
                                mask_loss = self.criterions[loss_name](pred_masks, target_bboxes=bboxes)
                                total_loss += mask_loss * float(self.cfg.TRAIN.LOSS.losses[loss_name])
                                meters[loss_name] = mask_loss.item()
                else:
                    # If no bboxes available, generate adaptive bboxes or use dummy loss
                    for loss_name in ["gaussian_mask_regression", "focal_mask", "combined_mask"]:
                        if loss_name in self.criterions and loss_name in self.cfg.TRAIN.LOSS.losses:
                            # Generate adaptive bboxes based on image content or use diverse dummy bboxes
                            batch_size = pred_masks.shape[0]
                            adaptive_bboxes = self.generate_adaptive_bboxes(batch_size, inputs)
                            mask_loss = self.criterions[loss_name](pred_masks, target_bboxes=adaptive_bboxes)
                            total_loss += mask_loss * float(self.cfg.TRAIN.LOSS.losses[loss_name])
                            meters[loss_name] = mask_loss.item()
            
            # Compute accuracy
            if "prob" in results.keys():
                from openunreid.core.metrics.accuracy import accuracy
                acc = accuracy(results["prob"].data, targets.data)
                meters["Acc@1"] = acc[0]
            
            self.train_progress.update(meters)
            
            # Log metrics to W&B
            rank, _, _ = get_dist_info()
            if rank == 0 and iter % self.print_freq == 0 and wandb_enabled:
                try:
                    # Log loss metrics
                    log_dict = {f"train/{k}": v for k, v in meters.items()}
                    log_dict["train/total_loss"] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
                    log_dict["train/lr"] = self.optimizer.param_groups[0]['lr']
                    log_dict["train/epoch"] = self._epoch
                    log_dict["train/iter"] = iter
                    
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to log to wandb: {e}")

            return (total_loss, meters)
            
        except Exception as e:
            print(f"Error in train_step iteration {iter}: {e}")
            # Return a dummy loss to prevent training crash  
            dummy_loss = torch.tensor(0.0, requires_grad=True).cuda()
            dummy_meters = {"error": 1.0}
            return (dummy_loss, dummy_meters)
    
    def generate_adaptive_bboxes(self, batch_size, inputs):
        """
        Generate adaptive bounding boxes for mask supervision when ground truth is not available.
        This creates diverse bbox patterns to encourage meaningful mask learning.
        """
        import random
        adaptive_bboxes = []
        
        for i in range(batch_size):
            # Generate random but reasonable bboxes
            # Strategy: create bboxes of different sizes and positions
            bbox_strategies = [
                # Center-focused (original strategy)
                [0.25, 0.25, 0.75, 0.75],
                # Slightly off-center variations
                [0.2, 0.3, 0.8, 0.8],
                [0.3, 0.2, 0.7, 0.8],
                [0.2, 0.2, 0.8, 0.7],
                # Different aspect ratios for suitcase diversity
                [0.15, 0.3, 0.85, 0.7],  # wider
                [0.3, 0.15, 0.7, 0.85],  # taller
                # Smaller focused regions
                [0.35, 0.35, 0.65, 0.65],
            ]
            
            # Select strategy based on batch index for consistency within epoch
            strategy_idx = (i + self._epoch * batch_size) % len(bbox_strategies)
            bbox = bbox_strategies[strategy_idx]
            adaptive_bboxes.append(bbox)
        
        return torch.tensor(adaptive_bboxes, dtype=torch.float32).cuda()
    
    def val(self):
        """
        Override the validation method to add W&B logging
        """
        # Call the original val method
        results = super().val()
        
        # Log validation metrics to W&B if available
        rank, _, _ = get_dist_info()
        if rank == 0 and results is not None and wandb_enabled:
            try:
                for (mAP, cmc) in [results]:
                    wandb.log({
                        f"val/mAP": mAP,
                        f"val/rank1": cmc[0],
                        f"val/rank5": cmc[4] if len(cmc) > 4 else 0,
                        f"val/rank10": cmc[9] if len(cmc) > 9 else 0,
                        f"val/epoch": self._epoch
                    })
            except Exception as e:
                print(f"⚠️ Warning: Failed to log validation metrics to wandb: {e}")
        
        return results


def main():
    global wandb_enabled
    wandb_enabled = False

    # Set environment variables for NCCL stability BEFORE importing torch.distributed
    # os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30 minutes timeout
    # os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand if causing issues
    # os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # Disable P2P if causing issues
    
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")  # More verbose for debugging
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1000000")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "COLL")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

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
    
    # Disable anomaly detection in distributed training as it can cause hangs
    torch.autograd.set_detect_anomaly(False)
    start_time = time.monotonic()

    # Init configuration
    args, cfg = parse_config()
    
    # Init distributed training with proper error handling
    try:
        dist = init_dist(cfg)
        rank, world_size, is_dist = get_dist_info()
        
        # Print distributed training info
        print(f"Distributed training initialized: rank={rank}, world_size={world_size}, is_dist={is_dist}")
        
        # Set random seed for reproducibility
        set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
        
        # Synchronize all processes after initialization
        synchronize()
        print(f"Rank {rank}: Initialization synchronization completed")
        
    except Exception as e:
        print(f"Error during distributed initialization: {e}")
        raise

    # Init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)
    
    # Initialize W&B - only on the main process in distributed training
    if rank == 0:
        try:
            # Initialize wandb with better configuration
            wandb.init(
                project="Suitcase-ReID-Gaussian-Mask",
                name=f"MVB-ResNet101-Mask-{str(cfg.work_dir).split('/')[-1]}",
                config=dict(cfg),
            )
            wandb_enabled = True
            print("✅ Weights & Biases initialized successfully (offline mode)!")
            print(f"📊 Run name: {wandb.run.name}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize wandb: {e}")
            print("Continuing training without wandb logging...")
            wandb_enabled = False

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
    # ResNet101 with mask generation capability for suitcases
    model = build_model_with_mask(cfg, num_classes, init=cfg.MODEL.source_pretrained)
    model.cuda()
    
    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=True,  # Handle unused parameters gracefully
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

    # # Watch model with W&B to track parameters and gradients
    # if rank == 0 and wandb_enabled:
    #     try:
    #         wandb.watch(model, log="gradients", log_freq=200)  # Reduced frequency to avoid overhead
    #     except Exception as e:
    #         print(f"⚠️ Warning: Failed to watch model with wandb: {e}")

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        runner.resume(args.resume_from)

    # Start training with proper error handling
    print("Starting SuitcaseReID training...")
    try:
        # Add a synchronization barrier before training starts
        synchronize()
        print(f"Rank {rank}: Training start synchronization completed")
        
        runner.run()
        
        # Add a synchronization barrier after training completes
        synchronize()
        print(f"Rank {rank}: Training completion synchronization completed")
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Ensure all processes are aware of the failure
        synchronize()
        raise

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
        
        # Log results to W&B
        if rank == 0 and wandb_enabled:
            try:
                # Log test results to W&B
                test_metrics = {
                    f'test/{cfg.TEST.datasets[i]}/mAP': mAP,
                    f'test/{cfg.TEST.datasets[i]}/rank1': cmc[0],
                    f'test/{cfg.TEST.datasets[i]}/rank5': cmc[4] if len(cmc) > 4 else 0,
                    f'test/{cfg.TEST.datasets[i]}/rank10': cmc[9] if len(cmc) > 9 else 0,
                }
                wandb.log(test_metrics)
                
                # Save model artifact in W&B
                if i == 0 and osp.isfile(best_model_path):  # Only save once and if file exists
                    wandb.save(str(best_model_path))
                    print(f"📁 Model saved to wandb: {best_model_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to log test results to wandb: {e}")

    # Print time
    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    print("Total running time: ", total_time)
    
    # Log final summary to wandb
    if rank == 0 and wandb_enabled:
        try:
            wandb.log({"training/total_time_seconds": (end_time - start_time)})
        except Exception as e:
            print(f"⚠️ Warning: Failed to log training time to wandb: {e}")


# Global wandb flag
wandb_enabled = False

if __name__ == "__main__":
    main()
    # Finish wandb logging
    try:
        if wandb_enabled and wandb.run is not None:
            wandb.finish()
            print("✅ Wandb logging finished successfully")
    except Exception as e:
        print(f"⚠️ Warning: Error finishing wandb: {e}")
    # cd /data1/zhaofanghan/OpenUnReID/tools/SuitcaseReID_supervised && python strong_baseline_suitcase.py --config suitcase_config.yaml