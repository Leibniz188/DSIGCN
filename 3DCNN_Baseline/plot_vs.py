"""
Description: Performance comparison plot (Error Violin Grid) between 3D-CNN 
             and DSIGCN for material property prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ==========================================
# ⚙️ Path Configurations
# ==========================================

# Unified relative asset paths matching preceding codebase pipelines
FILE_CNN = os.path.join('../Result/3dcnn/test_set_predictions.csv')
FILE_OURS = os.path.join('../Result/dsigcn/Predict/predict_example/prediction_values.csv')
SAVE_DIR = os.path.join('../Result/Model_Comparison_Final')

# ==========================================
# 🎨 Global Typography & Color Palettes (Minimalist)
# ==========================================
A4_WIDTH = 8.27

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "axes.grid": False,
    "figure.dpi": 500
})

COLORS = {
    'cnn': {'fill': '#72a4b8', 'line': '#5a8292'},
    'ours': {'fill': '#f4968d', 'line': '#c37871'},
    'ideal': '#95a5a6'
}


# ==========================================
# Utility Functions
# ==========================================
def get_tex_label(name):
    """Returns standardized LaTeX labels: variables bolded, units in normal font."""
    if 'v' in name:
        return r"$\mathbf{\nu}_{\mathbf{" + name[1:] + r"}}$"
    else:
        return r"$\mathbf{" + name[0] + r"}_{\mathbf{" + name[1:] + r"}}$ (GPa)"


def add_minimal_box_clean(ax, data, x_pos, color_config):
    """Draws a minimalist boxplot capturing median, quartiles, and whiskers."""
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    low_bound, high_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    valid_low = data[data >= low_bound].min() if len(data[data >= low_bound]) > 0 else np.min(data)
    valid_high = data[data <= high_bound].max() if len(data[data <= high_bound]) > 0 else np.max(data)

    box_color = 'k'
    whisker_color = color_config['line']

    ax.vlines(x_pos, valid_low, valid_high, colors=whisker_color, lw=1.0, alpha=0.8, zorder=4)
    w = 0.14  
    rect = plt.Rectangle((x_pos - w / 2, q1), w, q3 - q1,
                         edgecolor=box_color, facecolor='none', lw=0.6, zorder=5)
    ax.add_patch(rect)
    ax.hlines(med, x_pos - w / 2, x_pos + w / 2, colors=box_color, lw=1.2, zorder=6)


def plot_complete_violin_for_error(ax, pos, error_data, color_config):
    """Renders a complete standalone violin density profile for absolute errors."""
    y_min, y_max = np.min(error_data), np.max(error_data)
    buffer = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
    y_grid = np.linspace(y_min - buffer, y_max + buffer, 200)

    try:
        kde = gaussian_kde(error_data)(y_grid)
    except np.linalg.LinAlgError:
        ax.vlines(pos, y_min, y_max, colors=color_config['line'], lw=1.5)
        return

    # Normalize maximum distribution width limits to 0.4
    kde = kde / np.max(kde) * 0.4

    ax.fill_betweenx(y_grid, pos - kde, pos + kde, color=color_config['fill'], alpha=0.5, edgecolor='none', zorder=2)
    ax.plot(pos - kde, y_grid, color=color_config['line'], lw=1.3, zorder=3)
    ax.plot(pos + kde, y_grid, color=color_config['line'], lw=1.3, zorder=3)


# ==========================================
# Core Plotting Logic
# ==========================================
def plot_side_by_side_uq_error_violin_grid(df_cnn, df_ours, param_names, save_path, rows=3, cols=4):
    print(f"  > Plotting {rows}x{cols} error violin distribution grid...")

    # Layout bounds optimization
    left, right, bottom, top = 0.12, 0.98, 0.12, 0.96
    wspace, hspace = 0.35, 0.25

    usable_width = A4_WIDTH * (right - left)

    # Force subplots configuration ratio to 3:2 (Width to Height)
    w_sub = usable_width / (cols + (cols - 1) * wspace)
    h_sub = w_sub * (2 / 3)  
    usable_height = rows * h_sub + (rows - 1) * hspace * h_sub
    fig_height = usable_height / (top - bottom)

    fig, axes = plt.subplots(rows, cols, figsize=(A4_WIDTH, fig_height))
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    axes = axes.flatten()

    for i, prop_name in enumerate(param_names):
        ax = axes[i]

        true_data = df_ours[f"True_{prop_name}"].values
        pred_cnn = df_cnn[f"Pred_{prop_name}"].values
        pred_ours = df_ours[f"Pred_{prop_name}"].values

        mask = ~np.isnan(true_data) & ~np.isnan(pred_cnn) & ~np.isnan(pred_ours)
        true_data = true_data[mask]

        err_cnn = pred_cnn[mask] - true_data
        err_ours = pred_ours[mask] - true_data

        ax.axhline(0, color=COLORS['ideal'], linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)

        pos_cnn, pos_ours = 1, 2
        plot_complete_violin_for_error(ax, pos_cnn, err_cnn, COLORS['cnn'])
        plot_complete_violin_for_error(ax, pos_ours, err_ours, COLORS['ours'])

        add_minimal_box_clean(ax, err_cnn, pos_cnn, COLORS['cnn'])
        add_minimal_box_clean(ax, err_ours, pos_ours, COLORS['ours'])

        ax.set_xlim(0.4, 2.6)

        # Remove x-axis tick strings to retain minimalist theme
        ax.set_xticks([pos_cnn, pos_ours])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

        max_err = max(np.abs(err_cnn).max(), np.abs(err_ours).max())
        ax.set_ylim(-max_err * 1.1, max_err * 1.1)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(True, axis='y', ls=':', alpha=0.5, color='gray', zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Dynamic parameter labeling at top right corner
        label_tex = get_tex_label(prop_name)
        ax.text(0.96, 0.96, label_tex, transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.8, alpha=0.7), zorder=8)

    # Master y-axis label
    fig.text(0.02, 0.5, 'Prediction Error (Predicted - True)', va='center', rotation='vertical', fontsize=12)

    # Universal legend definitions
    cnn_patch = mpatches.Patch(color=COLORS['cnn']['fill'], alpha=0.6, label='3D-CNN Error Dist.')
    ours_patch = mpatches.Patch(color=COLORS['ours']['fill'], alpha=0.6, label='DSIGCN Error Dist.')

    fig.legend(handles=[cnn_patch, ours_patch], loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11, handlelength=1.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"✅ Figures successfully saved to: {save_path}")


# ==========================================
# Main Execution Entry
# ==========================================
def main():
    if not os.path.exists(FILE_CNN) or not os.path.exists(FILE_OURS):
        print("🚨 Error: Input CSV target files are missing configuration. Check dataset placement.")
        print(f"   Missing path 1 (CNN): {FILE_CNN}")
        print(f"   Missing path 2 (Ours): {FILE_OURS}")
        return

    df_cnn = pd.read_csv(FILE_CNN)
    df_ours = pd.read_csv(FILE_OURS)
    param_names = ['E11', 'E22', 'E33', 'G23', 'G13', 'G12', 'v12', 'v13', 'v21', 'v23', 'v31', 'v32']

    save_path_3x4 = os.path.join(SAVE_DIR, "Error_Violins_SideBySide_Clean_3x4.png")
    plot_side_by_side_uq_error_violin_grid(df_cnn, df_ours, param_names, save_path_3x4, rows=3, cols=4)


if __name__ == "__main__":
    main()