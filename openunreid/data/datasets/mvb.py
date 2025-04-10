#!/usr/bin/env python
"""
Dataset class for MVB - Modified for supervised training
"""

import glob
import os.path as osp
import re
import warnings
import numpy as np
from collections import defaultdict

from ..utils.base_dataset import ImageDataset


class MVB(ImageDataset):
    """MVB.
    A dataset for MVB suitcase re-identification with labeled data.
    
    Dataset structure:
    - MVB/
        - bounding_box_train/
        - bounding_box_test/
        - query/
    """

    dataset_dir = "MVB"

    def __init__(self, root, mode, val_split=0.2, del_labels=False, pseudo_labels=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        self.pseudo_labels = pseudo_labels
        assert (val_split > 0.0) and (
            val_split < 1.0
        ), "the percentage of val_set should be within (0.0,1.0)"

        subsets_cfgs = {
            "train": (
                osp.join(self.dataset_dir, "bounding_box_train"),
                [0.0, 1.0 - val_split],
                True,
            ),
            "val": (
                osp.join(self.dataset_dir, "bounding_box_train"),
                [1.0 - val_split, 1.0],
                False,
            ),
            "trainval": (
                osp.join(self.dataset_dir, "bounding_box_train"),
                [0.0, 1.0],
                True,
            ),
            "query": (osp.join(self.dataset_dir, "query"), [0.0, 1.0], False),
            "gallery": (
                osp.join(self.dataset_dir, "bounding_box_test"),
                [0.0, 1.0],
                False,
            ),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(mode)
            )

        required_files = [self.dataset_dir, cfgs[0]]
        self.check_before_run(required_files)

        self.mode = mode
        data = self.process_dir(*cfgs)
        super(MVB, self).__init__(data, mode, **kwargs)

    def process_dir(self, dir_path, data_range, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        
        # Sort to ensure deterministic behavior
        img_paths = sorted(img_paths)
        
        # For train/val split
        total_imgs = len(img_paths)
        if total_imgs == 0:
            raise RuntimeError(f"Found 0 images in directory: {dir_path}. Please check if the dataset path is correct.")
            
        start_idx = int(round(total_imgs * data_range[0]))
        end_idx = int(round(total_imgs * data_range[1]))
        img_paths = img_paths[start_idx:end_idx]
        
        # Try multiple pattern matching strategies for flexibility
        # 1. For filenames like "0001_c1s1_000123_01.jpg" (Market-1501 style)
        market_pattern = re.compile(r'([-\d]+)_')
        
        # 2. For filenames like "0010_p_3_0.jpg" (MVB style)
        suitcase_pattern = re.compile(r'(\d+)_[pg]_(\d+)\.jpg')
        
        # Extract person IDs from filenames
        pid_container = set()
        for img_path in img_paths:
            filename = osp.basename(img_path)
            
            # Try MVB pattern first
            pid_match = suitcase_pattern.search(filename)
            if pid_match:
                pid = int(pid_match.group(1))
                pid_container.add(pid)
                continue
                
            # Fall back to Market-1501 pattern
            pid_match = market_pattern.search(filename)
            if pid_match:
                pid = int(pid_match.group(1))
                pid_container.add(pid)
                continue
                
            # If no pattern matches, use a simple approach: just use the first numbers in the filename
            numbers = re.findall(r'\d+', filename)
            if numbers:
                pid = int(numbers[0])
                pid_container.add(pid)
        
        if len(pid_container) == 0:
            warnings.warn(f"No valid person IDs found in {dir_path}. Using sequential IDs instead.")
            # Assign sequential IDs if no proper IDs found
            data = []
            for i, img_path in enumerate(img_paths):
                pid = i // 4  # Assuming 4 images per ID on average
                camid = 0
                data.append((img_path, pid, camid))
            return data
                
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        data = []
        
        # Check if we're using pseudo labels from clustering (for semi-supervised)
        if self.pseudo_labels is not None:
            # Process with cluster-based pseudo-labels
            for i, (img_path, label) in enumerate(zip(img_paths, self.pseudo_labels)):
                pid = label  # Use the cluster ID as person ID
                if pid == -1:  # Handle outliers
                    if not self.mode.startswith('train'):
                        continue  # Skip outliers for validation
                    pid = len(img_paths) + i  # Give outliers unique IDs
                camid = 0  # Default camera ID
                data.append((img_path, pid, camid))
        else:
            # Process with real labels from filenames
            for img_path in img_paths:
                filename = osp.basename(img_path)
                
                # Try MVB pattern first (e.g. 0010_p_3_0.jpg)
                pid_match = suitcase_pattern.search(filename)
                if pid_match:
                    pid = int(pid_match.group(1))
                    view = int(pid_match.group(2))  # Use view ID as camera ID
                    
                    if self.del_labels:
                        pid = 0
                    else:
                        if relabel:
                            pid = pid2label[pid]
                    
                    data.append((img_path, pid, view))
                    continue
                
                # Try Market-1501 pattern
                pid_match = market_pattern.search(filename)
                if pid_match:
                    pid = int(pid_match.group(1))
                    
                    if self.del_labels:
                        pid = 0
                    else:
                        if relabel:
                            pid = pid2label[pid]
                            
                    # Extract camera ID if present, otherwise default to 0
                    cam_match = re.search(r'c(\d+)', filename)
                    camid = int(cam_match.group(1)) - 1 if cam_match else 0
                    
                    data.append((img_path, pid, camid))
                    continue
                    
                # If no pattern matches, use simple approach
                numbers = re.findall(r'\d+', filename)
                if numbers:
                    pid = int(numbers[0])
                    if self.del_labels:
                        pid = 0
                    else:
                        if pid in pid2label:  # Make sure the pid exists in our mapping
                            if relabel:
                                pid = pid2label[pid]
                        else:
                            warnings.warn(f"ID {pid} from {filename} not found in ID container. Using as-is.")
                    
                    camid = 0  # Default camera ID
                    data.append((img_path, pid, camid))
                else:
                    warnings.warn(f'No person ID could be extracted from {filename}, skipping.')
        
        return data
        
    def renew_labels(self, pseudo_labels):
        """Renew labels for the dataset with clustering results."""
        # This method is called after each clustering iteration in semi-supervised learning
        self.pseudo_labels = pseudo_labels
        
        # Re-process the data with new labels
        cfgs = {
            "train": (
                osp.join(self.dataset_dir, "bounding_box_train"),
                [0.0, 0.8],
                True,
            ),
            "val": (
                osp.join(self.dataset_dir, "bounding_box_train"),
                [0.8, 1.0],
                False,
            ),
            "trainval": (
                osp.join(self.dataset_dir, "bounding_box_train"),
                [0.0, 1.0],
                True,
            ),
        }[self.mode]
        
        self.data = self.process_dir(*cfgs)
        self.num_pids, self.num_cams = self.parse_data(self.data)