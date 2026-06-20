"""
Integrated Data Augmentation for RVE Dataset
Supports:
1. On-the-fly data augmentation (during training)
2. Offline batch augmentation (preprocessing)
"""

import os
import torch
import dgl
import numpy as np
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
import random

# Import data augmentation module
from data_augmentation import augment_rve_sample


class RVEDatasetWithAugmentation(DGLDataset):
    def __init__(self, bin_dir, force_normalize=True,
                 augment=False, aug_prob=0.5,
                 aug_axes=['z'], aug_angles=[90, 180, 270]):
        """
        Args:
            bin_dir (str): Directory path containing .bin files
            force_normalize (bool): Whether to perform Z-Score normalization on features and labels
            augment (bool): Whether to enable on-the-fly data augmentation
            aug_prob (float): Probability of applying data augmentation
            aug_axes (list): List of rotation axes, e.g., ['x', 'y', 'z']
            aug_angles (list): List of rotation angles in degrees, e.g., [90, 180, 270]
        """
        self.bin_dir = bin_dir
        self.force_normalize = force_normalize
        self.augment = augment
        self.aug_prob = aug_prob
        self.aug_axes = aug_axes
        self.aug_angles = aug_angles

        # Get all .bin files
        self.bin_files = [f for f in os.listdir(bin_dir) if f.endswith('.bin')]

        # Sort files numerically
        try:
            self.bin_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        except Exception as e:
            print(f"Warning: Could not sort files numerically. Using default sort. Error: {e}")
            self.bin_files.sort()

        # Statistics container
        self.stats = {
            'label': {'mean': None, 'std': None},
            'micro_node': {'mean': None, 'std': None},
            'micro_edge': {'mean': None, 'std': None},
            'meso_node': {'mean': None, 'std': None},
            'meso_edge': {'mean': None, 'std': None},
        }

        if self.force_normalize:
            self._compute_all_stats()

        super().__init__(name='RVE_Dual_View_Augmented')

    def _compute_all_stats(self):
        """
        Scan the entire dataset to compute Mean/Std for all features (Micro/Meso Node/Edge) and labels.
        Note: Augmented data is not used for computing statistics.
        """
        print("Pre-scanning dataset to compute GLOBAL feature statistics...")

        # Temporary lists to collect all data
        data_collector = {
            'label': [],
            'micro_node': [],
            'micro_edge': [],
            'meso_node': [],
            'meso_edge': []
        }

        for i, f in enumerate(self.bin_files):
            bin_path = os.path.join(self.bin_dir, f)
            gs, label_dict = dgl.load_graphs(bin_path)
            g_micro, g_meso = gs[0], gs[1]
            label = label_dict['label']

            # Collect Labels
            data_collector['label'].append(label)

            # Collect Micro features
            data_collector['micro_node'].append(g_micro.ndata['feat'])
            data_collector['micro_edge'].append(g_micro.edata['feat'])

            # Collect Meso features
            data_collector['meso_node'].append(g_meso.ndata['feat'])
            if g_meso.num_edges() > 0:
                data_collector['meso_edge'].append(g_meso.edata['feat'])

            # Simple progress print
            if (i + 1) % 50 == 0:
                print(f"  Scanned {i + 1}/{len(self.bin_files)} files...")

        print("Computing statistics...")

        # Compute Mean and Std
        for key in self.stats.keys():
            if len(data_collector[key]) > 0:
                all_data = torch.cat(data_collector[key], dim=0)

                mean = torch.mean(all_data, dim=0)
                std = torch.std(all_data, dim=0)

                # Prevent zero standard deviation
                std[std < 1e-6] = 1.0

                self.stats[key]['mean'] = mean
                self.stats[key]['std'] = std

                print(f"  [{key}] Dim: {mean.shape[0]} | Mean: {mean[:3].numpy()}... | Std: {std[:3].numpy()}...")
            else:
                print(f"  [{key}] No data found.")

    def process(self):
        pass

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        bin_path = os.path.join(self.bin_dir, self.bin_files[idx])
        gs, label_dict = dgl.load_graphs(bin_path)

        g_micro = gs[0]
        g_meso = gs[1]
        label = label_dict['label']  # Shape [1, 12]

        # === 🔥 Data Augmentation (Applied before normalization) ===
        if self.augment and self.training:
            if random.random() < self.aug_prob:
                # Randomly select rotation axis and angle
                axis = random.choice(self.aug_axes)
                angle = random.choice(self.aug_angles)

                g_micro, g_meso, label = augment_rve_sample(
                    g_micro, g_meso, label, axis=axis, angle_deg=angle
                )

        # === Normalization ===
        if self.force_normalize:
            # 1. Normalize Micro
            if self.stats['micro_node']['mean'] is not None:
                g_micro.ndata['feat'] = (g_micro.ndata['feat'] - self.stats['micro_node']['mean']) / \
                                        self.stats['micro_node']['std']

            if self.stats['micro_edge']['mean'] is not None:
                g_micro.edata['feat'] = (g_micro.edata['feat'] - self.stats['micro_edge']['mean']) / \
                                        self.stats['micro_edge']['std']

            # 2. Normalize Meso
            if self.stats['meso_node']['mean'] is not None:
                g_meso.ndata['feat'] = (g_meso.ndata['feat'] - self.stats['meso_node']['mean']) / \
                                       self.stats['meso_node']['std']

            if g_meso.num_edges() > 0 and self.stats['meso_edge']['mean'] is not None:
                g_meso.edata['feat'] = (g_meso.edata['feat'] - self.stats['meso_edge']['mean']) / \
                                       self.stats['meso_edge']['std']

            # 3. Normalize Label
            if self.stats['label']['mean'] is not None:
                label = (label - self.stats['label']['mean']) / self.stats['label']['std']

        return g_micro, g_meso, label

    def denormalize(self, normalized_label):
        """
        Denormalize the model predictions back to true physical values.
        """
        mean = self.stats['label']['mean']
        std = self.stats['label']['std']

        if not self.force_normalize or mean is None:
            return normalized_label

        device = normalized_label.device
        mean = mean.to(device)
        std = std.to(device)

        return normalized_label * std + mean

    @property
    def training(self):
        """Training mode flag (set externally)"""
        return getattr(self, '_training', False)

    def train(self):
        """Set to training mode"""
        self._training = True
        return self

    def eval(self):
        """Set to evaluation mode"""
        self._training = False
        return self


# collate_fn remains unchanged
def collate_fn(batch):
    micro_graphs, meso_graphs, labels = zip(*batch)

    # Correct meso_id offsets for batching
    offset_micro_graphs = []
    current_meso_offset = 0

    for i, g_micro in enumerate(micro_graphs):
        g_meso_curr = meso_graphs[i]
        num_meso_nodes = g_meso_curr.num_nodes()

        g_new = g_micro.clone()
        inc_ids = g_new.ndata['meso_id']
        mask = (inc_ids != -1)

        if current_meso_offset > 0:
            g_new.ndata['meso_id'][mask] = inc_ids[mask] + current_meso_offset

        offset_micro_graphs.append(g_new)
        current_meso_offset += num_meso_nodes

    batched_micro = dgl.batch(offset_micro_graphs)
    batched_meso = dgl.batch(meso_graphs)
    batched_labels = torch.cat(labels, dim=0)

    return batched_micro, batched_meso, batched_labels


# ==========================================
# Testing Code
# ==========================================
if __name__ == "__main__":
    TEST_BIN_DIR = "/home/leibo/Dual_data_particle300/"

    if os.path.exists(TEST_BIN_DIR):
        print("=" * 70)
        print("Testing Augmented Dataset")
        print("=" * 70)

        # Create dataset with augmentation support
        ds = RVEDatasetWithAugmentation(
            TEST_BIN_DIR,
            force_normalize=True,
            augment=True,
            aug_prob=1.0,  # 100% augmentation probability for testing
            aug_axes=['z'],
            aug_angles=[90]
        )

        # Set to training mode
        ds.train()

        # Get original sample
        print("\n[Original Sample]")
        ds.eval()
        g_micro_orig, g_meso_orig, label_orig = ds[0]
        print(f"Micro Node Features (First 3 dims - Centroid): {g_micro_orig.ndata['feat'][0, :3]}")
        print(f"Meso Node Features (First 3 dims - Center): {g_meso_orig.ndata['feat'][0, :3]}")

        # Get augmented sample
        print("\n[Augmented Sample - Rotated 90° around Z-axis]")
        ds.train()
        g_micro_aug, g_meso_aug, label_aug = ds[0]
        print(f"Micro Node Features (First 3 dims - Centroid): {g_micro_aug.ndata['feat'][0, :3]}")
        print(f"Meso Node Features (First 3 dims - Center): {g_meso_aug.ndata['feat'][0, :3]}")

        print("\n[Verification of Rotation]")
        print("Original centroid should be like: (x, y, z)")
        print("After rotation it should be like: (-y, x, z)")

        print("\n✅ Testing Complete")

    else:
        print("Path not found.")