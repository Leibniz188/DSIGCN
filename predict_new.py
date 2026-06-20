import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import norm, gaussian_kde
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.patches import FancyBboxPatch

# Import your custom modules
from Dataset_augmented import RVEDatasetWithAugmentation, collate_fn
from model import DualScaleGNN

# ==========================================
# 🎨 Global Layout and Color Configuration
# ==========================================
A4_WIDTH = 8.27

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": False,
    "figure.dpi": 300
})

COLORS = {
    'ref_fill': '#72a4b8', 'ref_line': '#5a8292',   # Reference: Teal-gray & matching dark lines
    'pred_fill': '#f4968d', 'pred_line': '#c37871',  # Prediction: Warm coral & matching dark lines
    'ideal_line': '#95a5a6', 'fit_line': '#C0392B',
    'density_low': '#1B2631', 'density_mid': '#1ABC9C', 'density_high': '#F1C40F',
}

cm_data = [COLORS['density_low'], COLORS['density_mid'], COLORS['density_high']]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("TealGlowHighContrast", cm_data, N=256)


class Config:
    # Note: The script is currently configured to run using the example folder. 
    # If you wish to use the full dataset, please change the path accordingly.
    
    TRAIN_BIN_DIR = os.path.join("./example_data/copped_fiber/BIN/")
    PREDICT_BIN_DIR = os.path.join("./example_data/copped_fiber/BIN/")
    EXP_DIR = os.path.join('./Result/dsigcn/')
    SAVE_DIR_root = os.path.join(EXP_DIR, "Predict")

    MODEL_PATH = os.path.join(EXP_DIR, "best_Ours_Aug.pth")
    SPLIT_INFO_PATH = os.path.join(EXP_DIR, "split_info.json")
    # Set to None to ignore split info and use START_INDEX/END_INDEX directly
    #SPLIT_INFO_PATH = None  
    START_INDEX = 0
    END_INDEX = 100

    HIDDEN_DIM = 48
    LAYERS = 3
    OUT_DIM = 12

    SAVE_DIR = os.path.join(SAVE_DIR_root, "predict_example")
    OUTPUT_CSV = os.path.join(SAVE_DIR, "prediction_values.csv")
    METRICS_CSV = os.path.join(SAVE_DIR, "metrics_summary.csv")
    FIG_DIR_UQ = os.path.join(SAVE_DIR, "Figures_UQ")
    FIG_DIR_SCATTER = os.path.join(SAVE_DIR, "Figures_Scatter")

    os.makedirs(SAVE_DIR, exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# Plotting Functions
# ==========================================

def get_tex_label(name):
    if 'v' in name:
        return f"$\\nu_{{{name[1:]}}}$"
    else:
        return f"${name[0]}_{{{name[1:]}}}$ (GPa)"


def plot_uq_histograms_unified(trues, preds, param_names, save_dir):
    print(f"  > Plotting UQ Histograms (2x6 layout, unified format, legend centered below)...")

    COLORS_DARK = {
        'ref_fill': '#4a7d94', 'ref_line': '#2e5f73',
        'pred_fill': '#e06b5f', 'pred_line': '#b84a3e',
    }

    cols, rows = 6, 2
    left, right, bottom, top = 0.06, 0.97, 0.2, 0.94
    wspace, hspace = 0.30, 0.25
    usable_width = A4_WIDTH * (right - left)

    w_sub = usable_width / (cols + (cols - 1) * wspace)
    h_sub = w_sub
    usable_height = rows * h_sub + (rows - 1) * hspace * h_sub
    fig_height = usable_height / (top - bottom)

    fig, axes = plt.subplots(rows, cols, figsize=(A4_WIDTH, fig_height))
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    axes = axes.flatten()

    from matplotlib.ticker import FuncFormatter

    def smart_fmt(val, pos):
        s = f'{val:.2f}'.rstrip('0').rstrip('.')
        return s

    stats_rows = []

    for i in range(12):
        ax = axes[i]
        ref_data, pred_data = trues[:, i], preds[:, i]
        name = param_names[i]

        mask = ~np.isnan(ref_data) & ~np.isnan(pred_data)
        ref_data, pred_data = ref_data[mask], pred_data[mask]

        mu_r, std_r = np.mean(ref_data), np.std(ref_data)
        mu_p, std_p = np.mean(pred_data), np.std(pred_data)

        stats_rows.append({
            'Property': name,
            'True_Mean': mu_r, 'True_Std': std_r,
            'Pred_Mean': mu_p, 'Pred_Std': std_p,
            'Rel_Error_Mean(%)': abs(mu_r - mu_p) / (abs(mu_r) + 1e-8) * 100,
        })

        vmin = min(ref_data.min(), pred_data.min())
        vmax = max(ref_data.max(), pred_data.max())
        bins = np.linspace(vmin, vmax, 30)
        bin_width = bins[1] - bins[0]

        ax.hist(ref_data, bins=bins, color=COLORS_DARK['ref_fill'], alpha=0.5, edgecolor='none')
        ax.hist(pred_data, bins=bins, color=COLORS_DARK['pred_fill'], alpha=0.5, edgecolor='none')

        x_rng = np.linspace(vmin, vmax, 200)
        ax.plot(x_rng, norm.pdf(x_rng, mu_r, std_r) * len(ref_data) * bin_width,
                color=COLORS_DARK['ref_line'], lw=2.0)
        ax.plot(x_rng, norm.pdf(x_rng, mu_p, std_p) * len(pred_data) * bin_width,
                color=COLORS_DARK['pred_line'], lw=2.0)

        label_tex = get_tex_label(name)
        ax.text(0.04, 0.95, label_tex, transform=ax.transAxes, fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8, lw=0.5), zorder=4)

        ax.xaxis.set_major_formatter(FuncFormatter(smart_fmt))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    import matplotlib.patches as mpatches
    ref_patch  = mpatches.Patch(color=COLORS_DARK['ref_fill'],  label='Reference',  alpha=0.8)
    pred_patch = mpatches.Patch(color=COLORS_DARK['pred_fill'], label='Prediction', alpha=0.8)
    fig.legend(handles=[ref_patch, pred_patch], loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False,
               fontsize=10)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "UQ_Histograms_Unified.png"), dpi=500, bbox_inches='tight')
    plt.close()

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(os.path.join(save_dir, "UQ_statistics.csv"), index=False)
    print(f"  > UQ statistics saved to {os.path.join(save_dir, 'UQ_statistics.csv')}")


def plot_3x4_compact_summary(trues, preds, param_names, save_path):
    print(f"  > Plotting 3x4 compact summary (scatter size adapted to 1/40, spacing x0.8)...")

    cols, rows = 4, 3
    left, right, bottom, top = 0.08, 0.90, 0.1, 0.96
    wspace, hspace = 0.36, 0.35
    usable_width = A4_WIDTH * (right - left)

    w_sub = usable_width / (cols + (cols - 1) * wspace)
    h_sub = w_sub  # Aspect ratio 1:1 (perfect square)
    usable_height = rows * h_sub + (rows - 1) * hspace * h_sub
    fig_height = usable_height / (top - bottom)

    s_size = (w_sub * 72 / 40) ** 2

    fig, axes = plt.subplots(rows, cols, figsize=(A4_WIDTH, fig_height))
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
    axes = axes.flatten()
    sc = None

    for i in range(12):
        ax = axes[i]
        x_data, y_data = trues[:, i], preds[:, i]
        name = param_names[i]
        xy = np.vstack([x_data, y_data])
        try:
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x_data[idx], y_data[idx], z[idx]
        except:
            x, y, z = x_data, y_data, np.ones_like(x_data)

        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        margin = (high - low) * 0.1
        ax.plot([low - margin, high + margin], [low - margin, high + margin], '--', c=COLORS['ideal_line'], lw=1.5,
                zorder=1)

        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, linestyle='--', color=COLORS['fit_line'], lw=2.0, alpha=0.9, zorder=2)

        sc = ax.scatter(x, y, c=z, s=s_size, cmap=CUSTOM_CMAP, alpha=0.9, edgecolors='none', zorder=3)

        label_tex = get_tex_label(name)
        ax.text(0.05, 0.95, label_tex, transform=ax.transAxes, fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8, lw=0.5), zorder=4)

        ax.set_aspect('equal', adjustable='box')
        ax.locator_params(axis='both', nbins=4)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        ax.grid(True, ls=':', alpha=0.8, zorder=0)

    fig.text(0.48, 0.02, 'Ground truth (FEA)', ha='center', fontsize=14)
    fig.text(0.015, 0.5, 'Prediction (DSIGCN)', va='center', rotation='vertical', fontsize=14)

    if sc is not None:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('KDE Density', fontsize=12)

    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def plot_single_sandwich(y_true, y_pred, name, save_path):
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = y_true[idx], y_pred[idx], z[idx]
    except:
        x, y, z = y_true, y_pred, np.ones_like(y_true)

    fig = plt.figure(figsize=(A4_WIDTH * 0.8, A4_WIDTH * 0.8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6],
                           wspace=0.03, hspace=0.03, left=0.12, right=0.85, bottom=0.12, top=0.95)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (high - low) * 0.05
    lims = [low - margin, high + margin]

    ax_main.plot(lims, lims, ls='--', c=COLORS['ideal_line'], lw=1.5, zorder=1, label='Ideal')
    m, b = np.polyfit(x, y, 1)
    ax_main.plot(x, m * x + b, color=COLORS['fit_line'], lw=2.5, linestyle='--', alpha=0.9, zorder=2, label='Fit')

    s_size = ((A4_WIDTH * 0.8 * 6 / 7) * 72 / 40) ** 2
    sc = ax_main.scatter(x, y, c=z, s=s_size, cmap=CUSTOM_CMAP, alpha=0.8, edgecolors='none', zorder=3)

    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.64])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('KDE Density', fontsize=12)

    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    text_str = f"$R^2 = {r2:.4f}$\n$\\mathrm{{MAPE}} = {mape:.2f}\\%$"
    ax_main.text(0.05, 0.95, text_str, transform=ax_main.transAxes, va='top', ha='left', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'), zorder=4)

    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)

    label_tex = get_tex_label(name)
    ax_main.set_xlabel(f"True {label_tex}", fontsize=12)
    ax_main.set_ylabel(f"Predicted {label_tex}", fontsize=12)

    ax_top.hist(y_true, bins=30, density=True, color=COLORS['ref_fill'], alpha=0.75, edgecolor='none')
    mu_t, std_t = norm.fit(y_true)
    x_range = np.linspace(lims[0], lims[1], 200)
    ax_top.plot(x_range, norm.pdf(x_range, mu_t, std_t), color=COLORS['ref_line'], lw=2)
    ax_top.axis('off')

    ax_right.hist(y_pred, bins=30, density=True, orientation='horizontal', color=COLORS['pred_fill'], alpha=0.75,
                  edgecolor='none')
    mu_p, std_p = norm.fit(y_pred)
    y_range = np.linspace(lims[0], lims[1], 200)
    ax_right.plot(norm.pdf(y_range, mu_p, std_p), y_range, color=COLORS['pred_line'], lw=2)
    ax_right.axis('off')

    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def main():
    cfg = Config()

    print("Step 1: Calculating Stats...")
    train_ds = RVEDatasetWithAugmentation(cfg.TRAIN_BIN_DIR, force_normalize=True, augment=False)
    stats = {}
    for k in train_ds.stats:
        stats[k] = {'mean': train_ds.stats[k]['mean'].clone().cpu(), 'std': train_ds.stats[k]['std'].clone().cpu()}
    del train_ds

    print("Step 2: Loading Prediction Data...")
    pred_ds = RVEDatasetWithAugmentation(cfg.PREDICT_BIN_DIR, force_normalize=False, augment=False)
    pred_ds.force_normalize = True
    pred_ds.stats = stats

    if cfg.SPLIT_INFO_PATH and os.path.exists(cfg.SPLIT_INFO_PATH):
        with open(cfg.SPLIT_INFO_PATH, 'r') as f:
            info = json.load(f)
        f_map = {f: i for i, f in enumerate(pred_ds.bin_files)}
        indices = [f_map[f] for f in info.get('test', []) if f in f_map]
    else:
        indices = list(range(cfg.START_INDEX, cfg.END_INDEX))

    # Guard against empty dataset if demo folder has fewer than END_INDEX files
    indices = [idx for idx in indices if idx < len(pred_ds.bin_files)]
    if not indices:
        print("⚠️ Warning: No valid sample indices found. Falling back to all available files.")
        indices = list(range(len(pred_ds.bin_files)))

    loader = DataLoader(Subset(pred_ds, indices), batch_size=32, collate_fn=collate_fn)

    print("Step 3: Inference...")
    s_micro, s_meso, _ = pred_ds[0]
    model = DualScaleGNN(
        micro_in_dim=s_micro.ndata['feat'].shape[1],
        micro_edge_dim=s_micro.edata['feat'].shape[1],
        meso_in_dim=s_meso.ndata['feat'].shape[1],
        meso_edge_dim=s_meso.edata['feat'].shape[1] if s_meso.num_edges() > 0 else 0,
        hidden_dim=cfg.HIDDEN_DIM, out_dim=cfg.OUT_DIM, layers=cfg.LAYERS
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for gm, gM, y in loader:
            gm, gM = gm.to(cfg.DEVICE), gM.to(cfg.DEVICE)
            p = pred_ds.denormalize(model(gm, gM)).cpu().numpy()
            t = pred_ds.denormalize(y).cpu().numpy()
            all_preds.append(p)
            all_trues.append(t)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    param_names = ['E11', 'E22', 'E33', 'G23', 'G13', 'G12', 'v12', 'v13', 'v21', 'v23', 'v31', 'v32']

    print("\nStep 4: Generating Figures...")
    os.makedirs(cfg.FIG_DIR_UQ, exist_ok=True)
    os.makedirs(cfg.FIG_DIR_SCATTER, exist_ok=True)

    plot_uq_histograms_unified(all_trues, all_preds, param_names, cfg.FIG_DIR_UQ)
    plot_3x4_compact_summary(all_trues, all_preds, param_names,
                             os.path.join(cfg.FIG_DIR_SCATTER, "Summary_Compact_3x4.png"))
    for i, name in enumerate(param_names):
        plot_single_sandwich(all_trues[:, i], all_preds[:, i], name,
                             os.path.join(cfg.FIG_DIR_SCATTER, f"Sandwich_{name}.png"))

    df = pd.DataFrame(np.hstack([all_trues, all_preds]),
                      columns=[f"True_{n}" for n in param_names] + [f"Pred_{n}" for n in param_names])
    df.insert(0, 'Sample_ID', indices)
    df.to_csv(cfg.OUTPUT_CSV, index=False)
    print(f"✅ Prediction values saved to {cfg.OUTPUT_CSV}")

    print("\nStep 6: Calculating and Saving Metrics...")
    metrics_data = []

    for i, name in enumerate(param_names):
        y_true = all_trues[:, i]
        y_pred = all_preds[:, i]
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        metrics_data.append({"Property": name, "R2": r2, "MAPE (%)": mape})

    df_metrics = pd.DataFrame(metrics_data)
    avg_r2 = df_metrics['R2'].mean()
    avg_mape = df_metrics['MAPE (%)'].mean()
    avg_row = pd.DataFrame([{'Property': 'Average', 'R2': avg_r2, 'MAPE (%)': avg_mape}])

    df_metrics = pd.concat([df_metrics, avg_row], ignore_index=True)
    df_metrics.to_csv(cfg.METRICS_CSV, index=False)

    print(f"✅ Metrics summary saved to {cfg.METRICS_CSV}")
    print("-" * 40)
    print(df_metrics.to_string(index=False))
    print("-" * 40)


if __name__ == "__main__":
    main()