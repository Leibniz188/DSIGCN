"""
@reference: https://github.com/Raocp/3D-ConvNeuralNet-material-property-prediction/blob/master/training%26plot.py
Description: This repo includes the dataset/code of 3D Convolutional Neural Network 
             for material property prediction.
"""

import os
import re
import time
import json
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ===========================================================
# 0. Environment & Path Configurations
# ===========================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Seamlessly connect to the local example_data directory structure
LABEL_CSV_PATH = os.path.join('../example_data/copped_fiber/csv/CF.csv')
PHASE_DATA_DIR = os.path.join('../example_data/copped_fiber/output_mat/')
SPLIT_JSON_PATH = os.path.join('../Result/dsigcn/split_info.json')

# Create output directories for saved models and results
os.makedirs(os.path.join('../Result/3dcnn/saved_models'), exist_ok=True)
MODEL_SAVE_PATH = os.path.join('../Result/3dcnn/saved_models/best_model.pth')
# ===========================================================


def Pred_vs_truth(Emin, Emax, Vmin, Vmax, y_pred, y_truth):
    """
    Plots the prediction vs ground truth comparison charts.
    Note: Column indices are hardcoded assuming the training order:
    E11, E22, E33, G23, G13, G12, v21, v31, v12, v32, v13, v23
    """
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12, 16))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    fig.text(0.5, 0.04, 'Prediction', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Ground truth', va='center', rotation='vertical', fontsize=20)

    # Helper function for scatter plots
    def plot_scatter(ax_obj, pred, truth, title, is_v=False):
        ax_obj.scatter(pred, truth, alpha=0.5, label=title, edgecolors='none', facecolor='blue', s=10)
        ax_obj.axis('square')
        if is_v:
            ax_obj.set_xlim([0.26, 0.34])
            ax_obj.set_ylim([0.26, 0.34])
        ax_obj.plot(ax_obj.get_xlim(), ax_obj.get_ylim(), label='Baseline', ls="-", c=".3", alpha=0.7)
        ax_obj.set_title(title)
        ax_obj.grid(True)
        ax_obj.legend()

    # Plot E and G (First 6 columns)
    titles_E_G = ['$E_{11}$', '$E_{22}$', '$E_{33}$', '$G_{23}$', '$G_{13}$', '$G_{12}$']
    for i in range(6):
        row = i // 3
        col = i % 3
        plot_scatter(ax[row, col], y_pred[:, i], y_truth[:, i], titles_E_G[i])

    # Plot v (Last 6 columns, ordered as v21, v31, v12, v32, v13, v23)
    titles_v = [r'$\nu_{21}$', r'$\nu_{31}$', r'$\nu_{12}$', r'$\nu_{32}$', r'$\nu_{13}$', r'$\nu_{23}$']
    for i in range(6):
        idx = i + 6
        row = (i + 6) // 3
        col = (i + 6) % 3
        plot_scatter(ax[row, col], y_pred[:, idx], y_truth[:, idx], titles_v[i], is_v=True)

    output_plot = os.path.join('../Result/3dcnn/Pred_vs_Truth.pdf')
    plt.savefig(output_plot, dpi=400)
    plt.close('all')


class CNN3D(nn.Module):
    """3D CNN Architecture"""

    def __init__(self, input_shape, num_classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self._to_linear = None
        self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *shape)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            self._to_linear = int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_x.size(0)
    return running_loss / len(data_loader.dataset)


def main():
    ########### 1. Load labels from CSV #############
    print(f"Loading labels from CSV: {LABEL_CSV_PATH}")
    if not os.path.exists(LABEL_CSV_PATH):
        raise FileNotFoundError(f"Label CSV file not found at: {LABEL_CSV_PATH}")

    df = pd.read_csv(LABEL_CSV_PATH, skipinitialspace=True)
    print(f"CSV shape: {df.shape}")
    if df.isnull().any().any():
        df = df.fillna(df.mean())

    # Enforce specific column extraction order to match evaluation requirements
    train_col_order = [
        'E11', 'E22', 'E33',
        'G23', 'G13', 'G12',
        'v21', 'v31', 'v12', 'v32', 'v13', 'v23'
    ]

    missing_cols = [col for col in train_col_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {missing_cols}")

    y_data = df[train_col_order].values
    _ = df['vf'].values  # Volume fraction tags

    print(f"Labels loaded in training order: {train_col_order}")
    print(f"y_data shape: {y_data.shape}")

    ########### 2. Load phase data from .mat files #############
    print(f"\nLoading inclusion phase matrices from: {PHASE_DATA_DIR}")
    
    # Compatible with both 'Voxel_*.mat' and 'phase_ellipsoid_*.mat' patterns
    mat_files = sorted(glob(os.path.join(PHASE_DATA_DIR, 'Voxel_*.mat')))
    
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No voxel .mat files found in {PHASE_DATA_DIR}")

    X_data_list = []
    for mat_file in mat_files:
        data = scipy.io.loadmat(mat_file)['phase']
        X_data_list.append(data)

    X_data = np.concatenate(X_data_list, axis=0)
    print(f"Total voxel samples loaded: {X_data.shape[0]}")

    if X_data.shape[0] != y_data.shape[0]:
        print(f"⚠ Alignment mismatch! Resizing samples from {X_data.shape[0]} and {y_data.shape[0]}")
        min_samples = min(X_data.shape[0], y_data.shape[0])
        X_data = X_data[:min_samples]
        y_data = y_data[:min_samples]

    # Reshape coordinates to standard PyTorch format (N, C, D, H, W)
    X_data = np.reshape(X_data, [X_data.shape[0], X_data.shape[1], X_data.shape[2], X_data.shape[3], 1])
    X_data = np.transpose(X_data, (0, 4, 1, 2, 3))

    ########### 3. JSON Split Logic #############
    print("\n" + "=" * 70)
    print(f"Loading Dataset Split from JSON: {SPLIT_JSON_PATH}")
    print("=" * 70)

    if not os.path.exists(SPLIT_JSON_PATH):
        raise FileNotFoundError(f"Split JSON file not found at: {SPLIT_JSON_PATH}")

    with open(SPLIT_JSON_PATH, 'r') as f:
        split_config = json.load(f)

    def parse_indices(file_list):
        indices = []
        for filename in file_list:
            try:
                # Extract sample index number from filename strings
                numbers = re.findall(r'\d+', filename)
                if numbers:
                    indices.append(int(numbers[0]))
            except Exception:
                pass
        # Convert to 0-indexed to align with Python arrays
        return np.array([idx - 1 for idx in indices if idx <= len(X_data)])

    train_idx = parse_indices(split_config['train'])
    val_idx = parse_indices(split_config['val'])
    test_idx = parse_indices(split_config['test'])

    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    print("=" * 70 + "\n")

    ########### 4. Scale data ############
    # Separately scale elastic/shear moduli (E_G, first 6 columns) and Poisson's ratios (v, last 6 columns)
    E_data = y_data[:, 0:6]
    scaler_E = MinMaxScaler(feature_range=(0, 1))
    scaler_E.fit(E_data)
    E_data_map = scaler_E.transform(E_data)

    v_data = y_data[:, 6:]
    scaler_v = MinMaxScaler(feature_range=(0, 1))
    scaler_v.fit(v_data)
    v_data_map = scaler_v.transform(v_data)

    y_data_map = np.concatenate([E_data_map, v_data_map], axis=1)

    # Dataset configurations and creation of DataLoaders
    input_shape = X_data.shape[2:]
    num_classes = y_data.shape[-1]
    batch_size = 25

    X_train = torch.FloatTensor(X_data[train_idx])
    y_train_map = torch.FloatTensor(y_data_map[train_idx])
    X_val = torch.FloatTensor(X_data[val_idx])
    y_val_map = torch.FloatTensor(y_data_map[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train_map), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_map), batch_size=batch_size, shuffle=False)

    ########### 5. Model & Training ############
    print("\nInitializing 3D CNN network architecture...")
    model = CNN3D(input_shape, num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss = []
    val_loss = []
    best_loss = 100.0
    epochs = 150

    print("\nStarting training pipeline...")
    start = time.time()

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 50)

        epoch_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_loss.append(epoch_train_loss)

        epoch_val_loss = evaluate(model, val_loader, criterion, device)
        val_loss.append(epoch_val_loss)

        print(f'Train Loss: {epoch_train_loss:.6f}')
        print(f'Val Loss:   {epoch_val_loss:.6f}')

        # Apply learning rate decay adjustments
        if epoch > 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00005

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'scaler_E': scaler_E,
                'scaler_v': scaler_v
            }, MODEL_SAVE_PATH)
            print(f'✓ Saved best model checkpoint to: {MODEL_SAVE_PATH}')

    print(f'\nTraining pipeline completed! Total Time: {(time.time() - start):.2f} sec')

    # Plot and save loss history curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(train_loss), 'b-', label='Train')
    plt.plot(np.array(val_loss), 'm-', label='Val.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    history_plot_path = os.path.join('../Result/3dcnn/training_history.png')
    plt.savefig(history_plot_path, dpi=300)

    ########### 6. Testing & CSV Output ############
    print("\nLoading the optimized model weights for evaluation...")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X_test = torch.FloatTensor(X_data[test_idx]).to(device)
    y_test = y_data[test_idx]

    with torch.no_grad():
        y_test_pred_map = model(X_test).cpu().numpy()

    # Inverse transformation to extract real-scale engineering constants
    E_test_pred = scaler_E.inverse_transform(y_test_pred_map[:, 0:6])
    v_test_pred = scaler_v.inverse_transform(y_test_pred_map[:, 6:])
    y_test_pred = np.concatenate([E_test_pred, v_test_pred], axis=1)

    # Compute Mean Absolute Relative Error (MARE)
    ARE = np.absolute(y_test_pred - y_test) / (y_test + 1e-8)
    MARE = np.mean(ARE, axis=0)
    print("\nMean Absolute Relative Error (MARE) [Internal Training Order]:")
    for i, name in enumerate(train_col_order):
        print(f'{name:4s}: {MARE[i] * 100:.2f}%')

    # Reorder parameters and export predictions to CSV file (restore natural physics order)
    print("\nFormatting and saving test set predictions...")
    reorder_idx = [0, 1, 2, 3, 4, 5, 8, 10, 6, 11, 7, 9]

    y_test_reordered = y_test[:, reorder_idx]
    y_pred_reordered = y_test_pred[:, reorder_idx]
    ids_data = (test_idx + 1).reshape(-1, 1)  # Convert back to 1-indexed format for storage

    final_data = np.hstack([ids_data, y_test_reordered, y_pred_reordered])

    header_names = [
        'Sample_ID',
        'True_E11', 'True_E22', 'True_E33', 'True_G23', 'True_G13', 'True_G12',
        'True_v12', 'True_v13', 'True_v21', 'True_v23', 'True_v31', 'True_v32',
        'Pred_E11', 'Pred_E22', 'Pred_E33', 'Pred_G23', 'Pred_G13', 'Pred_G12',
        'Pred_v12', 'Pred_v13', 'Pred_v21', 'Pred_v23', 'Pred_v31', 'Pred_v32'
    ]

    df_pred = pd.DataFrame(final_data, columns=header_names)
    df_pred['Sample_ID'] = df_pred['Sample_ID'].astype(int)
    
    output_csv_path = os.path.join('../Result/3dcnn/test_set_predictions.csv')
    df_pred.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"✓ Reordered evaluation CSV saved to: {output_csv_path}")

    print("\nGenerating prediction vs truth validation plots...")
    Pred_vs_truth(60, 140, 0.25, 0.35, y_test_pred, y_test)
    print(f"✓ Contrast figure saved to: {os.path.join('../Result/3dcnn/Pred_vs_Truth.pdf')}")

    print("\n" + "=" * 60)
    print("Pipeline Execution Success!")
    print("=" * 60)


if __name__ == '__main__':
    main()