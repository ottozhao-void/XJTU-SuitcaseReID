# Written for single-camera unlabeled suitcase dataset

import glob
import os.path as osp
import re
import warnings
import random
import numpy as np
from collections import defaultdict

from ..utils.base_dataset import ImageDataset


class SuitcaseReID(ImageDataset):
    """SuitcaseReID.
    A custom dataset for suitcase re-identification with single camera view.
    
    Dataset structure:
    - Suitcase-ReID/
        - bounding_box_train/
        - bounding_box_test/
        - query/
    """

    dataset_dir = "Suitcase-ReID"  # Your dataset directory name

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
        super(SuitcaseReID, self).__init__(data, mode, **kwargs)

    def process_dir(self, dir_path, data_range, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        
        # Sort to ensure deterministic behavior
        img_paths = sorted(img_paths)
        
        # For train/val split
        total_imgs = len(img_paths)
        start_idx = int(round(total_imgs * data_range[0]))
        end_idx = int(round(total_imgs * data_range[1]))
        img_paths = img_paths[start_idx:end_idx]
        
        data = []
        
        # Check if we have pseudo labels (from clustering)
        if self.pseudo_labels is not None:
            # Process with cluster-based pseudo-labels
            for i, (img_path, label) in enumerate(zip(img_paths, self.pseudo_labels)):
                pid = label  # Use the cluster ID as person ID
                if pid == -1:  # Handle outliers
                    if not self.mode.startswith('train'):
                        continue  # Skip outliers for validation
                    pid = len(img_paths) + i  # Give outliers unique IDs
                camid = 0  # Single camera
                data.append((img_path, pid, camid))
        else:
            if self.mode.startswith('train') and not self.del_labels:
                # For initial training without clusters, create artificial diversity
                # Group images by filename patterns to create initial "classes"
                pattern_groups = defaultdict(list)
                for img_path in img_paths:
                    # Extract patterns from filenames like "suitcase0-3404_suitcase_1000_0.jpg"
                    filename = osp.basename(img_path)
                    # Use the middle part as a rough grouping (e.g., "1000" from "suitcase_1000_0")
                    pattern = filename.split('_')[1] if len(filename.split('_')) > 1 else 'default'
                    pattern_groups[pattern].append(img_path)
                
                # Now create data with these pattern-based groups
                pid_container = sorted(pattern_groups.keys())
                pid2label = {pid: label for label, pid in enumerate(pid_container)}
                
                for pattern, paths in pattern_groups.items():
                    for img_path in paths:
                        if self.del_labels:
                            pid = 0
                        else:
                            pid = pid2label[pattern]
                        camid = 0
                        data.append((img_path, pid, camid))
            else:
                # For validation/testing or when del_labels is True
                for i, img_path in enumerate(img_paths):
                    pid = i if not self.del_labels else 0
                    camid = 0
                    data.append((img_path, pid, camid))
        
        return data
        
    def renew_labels(self, pseudo_labels):
        """Renew labels for the dataset with clustering results."""
        # This method is called after each clustering iteration
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