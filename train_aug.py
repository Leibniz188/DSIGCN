# train_aug.py (Modified Version - Runs two experiments: Augmentation + No Augmentation)
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import dgl
import matplotlib.pyplot as plt

from Dataset_augmented import RVEDatasetWithAugmentation, collate_fn
from model import DualScaleGNN

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


class Config:
    # Note: The script is currently configured to run using the example folder. 
    # If you wish to use the full dataset, please change the path accordingly.
    BIN_DIR = os.path.join('./example_data/copped_fiber/BIN/')
    OUTPUT_DIR = os.path.join('./Result/dsigcn/')
    
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    SPLIT_INFO_PATH = os.path.join(OUTPUT_DIR, "split_info.json")
    LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
    DIST_PLOT_PATH = os.path.join(OUTPUT_DIR, "split_distribution.png")

    # Data Augmentation Configuration
    AUG_PROB = 0.75
    AUG_AXES = ['x', 'y', 'z']
    AUG_ANGLES = [90, 180, 270]

    # Training Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
    # Mod 1: NoAug uses stronger regularization; Aug itself acts as regularization so it uses weaker weight_decay
    WEIGHT_DECAY_NOAUG = 1e-3
    WEIGHT_DECAY_AUG   = 1e-3
    EPOCHS = 500
    PATIENCE = 40
    PLOT_INTERVAL = 10
    SAVE_METRICS_INTERVAL = 5

    # Model Parameters
    HIDDEN_DIM = 48
    LAYERS = 3
    OUT_DIM = 12

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(Config.PLOT_DIR, exist_ok=True)


def plot_test_scatter(preds, targets, epoch, save_path):
    component_names = [f"C_{i + 1}" for i in range(12)]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Test Set Predictions - Epoch {epoch}', fontsize=20)

    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        y_pred = preds[:, i]
        y_true = targets[:, i]

        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.5, s=10)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax.set_title(f"{component_names[i]} (R2={r2:.4f})")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(save_path)
    plt.close()


def plot_data_distribution(all_vfs, train_idx, val_idx, test_idx, save_path):
    train_vfs = all_vfs[train_idx]
    val_vfs = all_vfs[val_idx]
    test_vfs = all_vfs[test_idx]

    bins = np.linspace(all_vfs.min(), all_vfs.max(), 20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Volume Fraction Distribution across Splits', fontsize=16)

    axes[0].hist(train_vfs, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Train Set (N={len(train_vfs)})')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    axes[1].hist(val_vfs, bins=bins, color='#ff7f0e', alpha=0.7, edgecolor='black')
    axes[1].set_title(f'Validation Set (N={len(val_vfs)})')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    axes[2].hist(test_vfs, bins=bins, color='#2ca02c', alpha=0.7, edgecolor='black')
    axes[2].set_title(f'Test Set (N={len(test_vfs)})')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('Volume Fraction')
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Distribution plot saved to {save_path}")


def perform_stratified_split(dataset, bin_dir):
    print("Extracting Volume Fractions for Stratified Splitting...")

    indices = np.arange(len(dataset))
    vfs = []

    for filename in dataset.bin_files:
        path = os.path.join(bin_dir, filename)
        gs, _ = dgl.load_graphs(path)
        g_meso = gs[1]

        if g_meso.ndata['feat'].shape[1] >= 16:
            total_vol = g_meso.ndata['feat'][:, 15].sum().item()
        else:
            total_vol = 0.0

        vfs.append(total_vol)

    vfs = np.array(vfs)
    print(f"Volume Fraction Stats: Min={vfs.min():.4f}, Max={vfs.max():.4f}, Mean={vfs.mean():.4f}")

    n_bins = 10 if len(dataset) > 50 else 3
    vf_bins = pd.cut(vfs, bins=n_bins, labels=False)

    unique, counts = np.unique(vf_bins, return_counts=True)
    if np.any(counts < 2):
        print("⚠️ Warning: Some bins have too few samples. Falling back to random split.")
        vf_bins = None

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=vf_bins,
        random_state=42,
        shuffle=True
    )

    if vf_bins is not None:
        temp_bins = vf_bins[temp_idx]
        unique_temp, counts_temp = np.unique(temp_bins, return_counts=True)
        stratify_temp = temp_bins if not np.any(counts_temp < 2) else None
    else:
        stratify_temp = None

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=stratify_temp,
        random_state=42,
        shuffle=True
    )

    return train_idx, val_idx, test_idx, vfs


def compute_r2_mape(preds, targets):
    r2_avg = r2_score(targets, preds)
    mape_overall = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    return r2_avg, mape_overall


def train_single_experiment(use_augmentation, train_idx, val_idx, test_idx,
                             cfg, shared_dataset):
    """
    Train a single experiment (with or without augmentation)
    Mod 2: Accepts shared_dataset to reuse its statistics, avoiding repeated scanning of the dataset
    """
    exp_name = "Ours_Aug" if use_augmentation else "Ours_NoAug"
    weight_decay = cfg.WEIGHT_DECAY_AUG if use_augmentation else cfg.WEIGHT_DECAY_NOAUG  # Mod 1

    print(f"\n{'#' * 70}")
    print(f"Experiment: {exp_name}")
    print(f"{'#' * 70}")

    if use_augmentation:
        print("\n" + "=" * 70)
        print("Data Augmentation Enabled")
        print(f"  Augmentation Probability: {cfg.AUG_PROB}")
        print(f"  Rotation Axes: {cfg.AUG_AXES}")
        print(f"  Rotation Angles: {cfg.AUG_ANGLES}")
        expected_multiplier = 1 + len(cfg.AUG_ANGLES) * cfg.AUG_PROB
        print(f"  Expected Training Set Expansion: {expected_multiplier:.2f}x")
        print(f"  Weight Decay: {weight_decay}")
        print("=" * 70 + "\n")
    else:
        print(f"\nData Augmentation Disabled | Weight Decay: {weight_decay}\n")

    # Mod 2: Reuse shared_dataset statistics, only toggle the augment flag without rescanning
    full_dataset = shared_dataset
    full_dataset.augment = use_augmentation
    full_dataset.aug_prob = cfg.AUG_PROB
    full_dataset.aug_axes = cfg.AUG_AXES
    full_dataset.aug_angles = cfg.AUG_ANGLES

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # Get feature dimensions
    full_dataset.eval()
    sample_micro, sample_meso, _ = train_set[0]

    micro_in_dim = sample_micro.ndata['feat'].shape[1]
    micro_edge_dim = sample_micro.edata['feat'].shape[1]
    meso_in_dim = sample_meso.ndata['feat'].shape[1]
    meso_edge_dim = sample_meso.edata['feat'].shape[1] if sample_meso.num_edges() > 0 else 0

    # Create model
    model = DualScaleGNN(
        micro_in_dim=micro_in_dim,
        micro_edge_dim=micro_edge_dim,
        meso_in_dim=meso_in_dim,
        meso_edge_dim=meso_edge_dim,
        hidden_dim=cfg.HIDDEN_DIM,
        out_dim=cfg.OUT_DIM,
        layers=cfg.LAYERS
    ).to(cfg.DEVICE)

    print(f"\nModel Architecture:")
    print(f"  Micro Input: Node={micro_in_dim}, Edge={micro_edge_dim}")
    print(f"  Meso Input: Node={meso_in_dim}, Edge={meso_edge_dim}")
    print(f"  Hidden Dim: {cfg.HIDDEN_DIM}")
    print(f"  Layers: {cfg.LAYERS}")
    print(f"  Output Dim: {cfg.OUT_DIM}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")

    if use_augmentation:
        effective_train_size = len(train_idx) * (1 + len(cfg.AUG_ANGLES) * cfg.AUG_PROB)
    else:
        effective_train_size = len(train_idx)
    param_sample_ratio = total_params / effective_train_size
    print(f"  Param/Sample Ratio: {param_sample_ratio:.1f}:1")

    # Optimizer: Use corresponding weight_decay for each experiment
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    metrics_history = []
    best_r2_avg = 0.0
    best_mape = float('inf')

    print("\nStart Training...")

    for epoch in range(cfg.EPOCHS):
        start_time = time.time()

        # Training
        model.train()
        if use_augmentation:
            full_dataset.train()
        else:
            full_dataset.eval()

        train_loss = 0.0
        for g_micro, g_meso, targets in train_loader:
            g_micro, g_meso, targets = g_micro.to(cfg.DEVICE), g_meso.to(cfg.DEVICE), targets.to(cfg.DEVICE)

            optimizer.zero_grad()
            preds = model(g_micro, g_meso)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        full_dataset.eval()

        val_loss = 0.0
        with torch.no_grad():
            for g_micro, g_meso, targets in val_loader:
                g_micro, g_meso, targets = g_micro.to(cfg.DEVICE), g_meso.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                preds = model(g_micro, g_meso)
                val_loss += criterion(preds, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        val_train_ratio = avg_val_loss / (avg_train_loss + 1e-8)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0

            model_save_path = os.path.join(cfg.OUTPUT_DIR, f"best_{exp_name}.pth")
            torch.save(model.state_dict(), model_save_path)
            save_msg = f"--> Best ({best_val_loss:.4f})"

            test_preds_list = []
            test_targets_list = []

            with torch.no_grad():
                for g_micro, g_meso, targets in test_loader:
                    g_micro, g_meso, targets = g_micro.to(cfg.DEVICE), g_meso.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                    preds = model(g_micro, g_meso)

                    preds_real = full_dataset.denormalize(preds)
                    targets_real = full_dataset.denormalize(targets)

                    test_preds_list.append(preds_real.cpu().numpy())
                    test_targets_list.append(targets_real.cpu().numpy())

            all_preds_real = np.concatenate(test_preds_list, axis=0)
            all_targets_real = np.concatenate(test_targets_list, axis=0)

            best_r2_avg, best_mape = compute_r2_mape(all_preds_real, all_targets_real)
        else:
            early_stop_counter += 1
            save_msg = ""

        # Save test metrics at specified intervals
        if (epoch + 1) % cfg.SAVE_METRICS_INTERVAL == 0:
            metrics_history.append({
                'Epoch': epoch + 1,
                'R2_avg': best_r2_avg,
                'MAPE': best_mape
            })

        # Periodic testing and plotting
        if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
            test_preds_list = []
            test_targets_list = []

            with torch.no_grad():
                for g_micro, g_meso, targets in test_loader:
                    g_micro, g_meso, targets = g_micro.to(cfg.DEVICE), g_meso.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                    preds = model(g_micro, g_meso)

                    preds_real = full_dataset.denormalize(preds)
                    targets_real = full_dataset.denormalize(targets)

                    test_preds_list.append(preds_real.cpu().numpy())
                    test_targets_list.append(targets_real.cpu().numpy())

            all_preds_real = np.concatenate(test_preds_list, axis=0)
            all_targets_real = np.concatenate(test_targets_list, axis=0)

            plot_dir = os.path.join(cfg.PLOT_DIR, exp_name)
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"scatter_epoch_{epoch + 1}.png")
            plot_test_scatter(all_preds_real, all_targets_real, epoch + 1, plot_path)

            test_r2 = r2_score(all_targets_real, all_preds_real)
            print(f"    [Test] Epoch {epoch + 1} | R2: {test_r2:.4f} | Plot saved")

        # Print logs
        epoch_time = time.time() - start_time

        if val_train_ratio > 2.0:
            overfitting_warning = " ⚠️ Overfitting"
        elif val_train_ratio > 1.5:
            overfitting_warning = " ⚠️"
        else:
            overfitting_warning = ""

        print(f"Epoch {epoch + 1:03d}/{cfg.EPOCHS} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Ratio: {val_train_ratio:.2f}{overfitting_warning} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s {save_msg}")

        # Early stopping
        if early_stop_counter >= cfg.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\n{exp_name} Training Finished.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    loss_curve_path = os.path.join(cfg.OUTPUT_DIR, f"{exp_name}_loss_curve.png")
    plot_loss_curve(history['train_loss'], history['val_loss'], loss_curve_path)
    print(f"Loss curve saved to {loss_curve_path}")

    if metrics_history:
        csv_path = os.path.join(cfg.OUTPUT_DIR, f"{exp_name}_metrics.csv")
        df = pd.DataFrame(metrics_history)
        df.to_csv(csv_path, index=False)
        print(f"✅ Metrics history saved to {csv_path}")

    return best_r2_avg, best_mape


def main():
    cfg = Config()
    print(f"Using device: {cfg.DEVICE}")

    print(f"\nLoading dataset from {cfg.BIN_DIR}...")

    # Mod 2: Scan only once, both experiments share the same statistics
    shared_dataset = RVEDatasetWithAugmentation(
        cfg.BIN_DIR,
        force_normalize=True,
        augment=False,
        aug_prob=cfg.AUG_PROB,
        aug_axes=cfg.AUG_AXES,
        aug_angles=cfg.AUG_ANGLES
    )

    train_idx, val_idx, test_idx, all_vfs = perform_stratified_split(shared_dataset, cfg.BIN_DIR)

    plot_data_distribution(all_vfs, train_idx, val_idx, test_idx, cfg.DIST_PLOT_PATH)

    split_info = {
        'train': [shared_dataset.bin_files[i] for i in train_idx],
        'val': [shared_dataset.bin_files[i] for i in val_idx],
        'test': [shared_dataset.bin_files[i] for i in test_idx]
    }
    with open(cfg.SPLIT_INFO_PATH, 'w') as f:
        json.dump(split_info, f, indent=4)
    print(f"Dataset split info saved to {cfg.SPLIT_INFO_PATH}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    results = {}

    # Experiment 1: No Augmentation
    r2_noaug, mape_noaug = train_single_experiment(
        use_augmentation=False,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cfg=cfg,
        shared_dataset=shared_dataset
    )
    results['Ours_NoAug'] = {'R2_avg': r2_noaug, 'MAPE': mape_noaug}

    # Experiment 2: With Augmentation
    r2_aug, mape_aug = train_single_experiment(
        use_augmentation=True,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cfg=cfg,
        shared_dataset=shared_dataset
    )
    results['Ours_Aug'] = {'R2_avg': r2_aug, 'MAPE': mape_aug}

    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Experiment':<20} {'R²_avg':>10} {'MAPE(%)':>10}")
    print("-" * 70)
    for exp_name, metrics in results.items():
        print(f"{exp_name:<20} {metrics['R2_avg']:>10.4f} {metrics['MAPE']:>9.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()