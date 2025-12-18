import math
import torch
import numpy as np
import os
from torch.utils.data import Sampler
import pandas as pd
from loguru import logger

class BucketBatchSampler(Sampler):
    def __init__(
        self,
        duration_file,
        batch_size,
        drop_last=False,
        bucket_boundaries=None,
        shuffle=True,
        seed=42
    ):
        """
        Sampler that buckets samples by duration to minimize padding.
        
        Args:
            duration_file (str or list): Path to duration.txt or list of durations.
                                         If path, expects one float per line.
            batch_size (int): Batch size.
            drop_last (bool): Whether to drop the last incomplete batch in each bucket.
            bucket_boundaries (list): List of float boundaries for buckets. 
                                      If None, uses a single bucket (random sampling).
            shuffle (bool): Whether to shuffle samples within buckets and shuffle bucket order.
            seed (int): Random seed.
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Load durations
        if isinstance(duration_file, str) and os.path.exists(duration_file):
            with open(duration_file, 'r') as f:
                self.durations = np.array([float(line.strip()) for line in f if line.strip()])
        elif isinstance(duration_file, (list, np.ndarray, pd.Series)):
             self.durations = np.array(duration_file)
        else:
            raise ValueError(f"Invalid duration_file: {duration_file}")

        self.num_samples = len(self.durations)
        
        # Validate/Process boundaries
        if bucket_boundaries is None:
            # Fallback to essentially random sampling if no boundaries provided
            self.bucket_boundaries = [min(self.durations), max(self.durations) + 0.1]
        else:
            self.bucket_boundaries = sorted(bucket_boundaries)
            # Ensure boundaries cover the range
            if self.bucket_boundaries[0] > min(self.durations):
                self.bucket_boundaries.insert(0, min(self.durations))
            if self.bucket_boundaries[-1] <= max(self.durations):
                self.bucket_boundaries.append(max(self.durations) + 0.01)

        self._assign_buckets()
        
    def _assign_buckets(self):
        """Assigns each sample index to a bucket."""
        # digitize returns 1-based indices, so we subtract 1
        # If value == boundary, it goes to the right bin usually, but we clip just in case
        self.bucket_indices = np.digitize(self.durations, self.bucket_boundaries) - 1
        
        # Clip to ensure valid range [0, len(boundaries)-2]
        # (len(boundaries) - 1 is the number of buckets)
        n_buckets = len(self.bucket_boundaries) - 1
        self.bucket_indices = np.clip(self.bucket_indices, 0, n_buckets - 1)
        
        self.buckets = {}
        for i in range(n_buckets):
            # Get indices falling into this bucket
            indices = np.where(self.bucket_indices == i)[0]
            if len(indices) > 0:
                self.buckets[i] = indices

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        batches = []
        
        # Keys (bucket IDs) to iterate over
        bucket_ids = list(self.buckets.keys())
        
        # Shuffle order of buckets if needed (optional, but good for mixing types of data)
        # However, usually we want to shuffle the batches *after* creation effectively, 
        # but iterating bucket-by-bucket is standard for minimizing padding.
        # Strict bucketing means we finish one bucket before moving to next? 
        # Usually yes, or we interleave batches from different buckets. 
        # The standard approach is:
        # 1. Create batches within each bucket.
        # 2. Collect ALL batches from ALL buckets.
        # 3. Shuffle the LIST of batches (so we mix short and long batches in the epoch).
        
        all_batches_Global = []

        for b_id in bucket_ids:
            indices = self.buckets[b_id]
            
            # Shuffle within bucket
            if self.shuffle:
                indices = indices[torch.randperm(len(indices), generator=g).numpy()]
            
            # Chunk into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                all_batches_Global.append(batch)
        
        # Shuffle the order of batches
        if self.shuffle:
            # We shuffle the list of batches
            batch_indices = torch.randperm(len(all_batches_Global), generator=g).tolist()
            all_batches_Global = [all_batches_Global[i] for i in batch_indices]
            
        for batch in all_batches_Global:
            yield batch

    def __len__(self):
        count = 0
        for b_id in self.buckets:
            indices = self.buckets[b_id]
            length = len(indices)
            if self.drop_last:
                count += length // self.batch_size
            else:
                count += (length + self.batch_size - 1) // self.batch_size
        return count

    def set_epoch(self, epoch):
        self.epoch = epoch
