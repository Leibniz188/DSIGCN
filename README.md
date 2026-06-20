# DSIGCN: Dual-Scale Interactive Graph Convolutional Neural Network & 3D-CNN Baseline

## Introduction

This project presents a **Dual-Scale Interactive Graph Convolutional Neural Network (DSIGCN)** for the homogenization of 3D Representative Volume Elements (RVEs) with random inclusions. Multi-scale graph representations are constructed from physical mesh data to predict the effective macroscopic mechanical properties of heterogeneous materials.

To provide a rigorous performance benchmark, this repository also includes a comprehensive **3D Convolutional Neural Network (3D-CNN)** baseline module based on voxelized phase data, complete with automated end-to-end evaluation and cross-model visualization utilities.

The codebase bridges the gap between traditional Finite Element Analysis (FEA), spatial volumetric networks, and modern Graph Neural Networks (GNNs).
---

## Project Structure

```text
DSIGCN/
│
├── example_data/                 # Mini dataset for immediate testing out-of-the-box
│   ├── raw_mesh/                 # Raw .dat FEA mesh files
│   ├── output_mat/               # Generated 3D voxel phase tensors (.mat)
│   ├── targets.csv               # Material properties and macroscopic labels
│   ├── split_info.json           # Shared train/val/test data index partition records
│   ├── prediction_values.csv     # DSIGCN target predictions history
│   └── Model_Comparison_Final/   # Saved evaluation grid visualizations
│
├── Pre_mesh2graph/
│   ├── mesh_data_extractor.py    # Parses raw FEA mesh files into structured text
│   └── Make_graph_bin.py         # Constructs Dual-Scale Graphs & Labels, saves as .bin
│
├── 3DCNN_Baseline/
│   ├── Generate_mat.py           # Voxelizes geometry coordinates into 3D .mat volumes
│   ├── train_3dcnn.py            # Main PyTorch 3D-CNN training and testing script
│   └── plot_vs.py                # Publication-ready error distribution violin comparison
│
├── Dataset_augmented.py          # Custom DGL dataset & tensor transformation logic
├── model.py                      # DualScaleGNN model architecture
├── train_aug.py                  # DSIGCN comparative training and evaluation script
│
├── test_set_predictions.csv      # Exported test predictions from 3D-CNN (auto-generated)
└── README.md

```

---

## Requirements

Our experiments are conducted on a server equipped with Ubuntu 20.04 (Kernel 5.15.0-139-generic), single Intel(R) Xeon(R) Platinum 8470 CPU, NVIDIA H100 PCIe 80GB GPUs, Driver 550.144.03, CUDA Toolkit 12.4 and Python 3.8.20. Install all dependencies via  `requirements.txt`  with  `pip install -r requirements.txt` ; PyTorch installation commands differ for different CUDA/GPU setups, please check the official PyTorch website for a matching command for your own machine.

This repository releases only the core implementation of our proposed method. All ablation experiments and out-of-distribution (OOD) tests reported in our paper can be reproduced by simple modifications to the provided codes. All datasets used in this work are publicly available.


```bash
pip install -r requirements.txt
```

---

## Dataset

Due to GitHub's file size limitations, the full training dataset is hosted externally.

* **Full Dataset:** [Insert Link to Zenodo / Hugging Face / Google Drive Here]
* **Example Data:** We provide a miniature dataset in the `example_data/` directory so you can immediately test preprocessing, graph construction, voxelization, and training workflows right after cloning the repository.

---

## Data Processing Pipelines

### 1. The `Pre_mesh2graph` Graph Pipeline

**Raw FEA Mesh (.dat) + Properties (.csv) → Structured Text → Training Graph Packages (.bin)**

* **Extraction (`mesh_data_extractor.py`):** Parses raw FEA element data (e.g., C3D4 tetrahedrals) via keyword parsing into clean topological texts: `node.txt`, `element.txt`, and `matrix.txt`.
* **Graph Synthesis (`Make_graph_bin.py`):** Constructs element-level **Micro Graphs** and inclusion-level **Meso Graphs** embedding Periodic Boundary Conditions (PBC) and PCA shape descriptors, packaging them into standalone DGL binary objects.

### 2. The `3DCNN_Baseline` Voxel Pipeline

**Geometry Metadata → Spatial Phase Voxelization (.mat) → Volumetric Deep Learning**

* **Generation (`Generate_mat.py`):** Discretizes continuous multi-phase RVE geometries into structured 3D grids (Voxel), saving inclusion/matrix layouts as binary spatial phase matrix fields inside `.mat` files.

---

## 🚀 Quick Start (Demo)

All scripts utilize robust script-relative path configurations (`os.path.dirname`). You can execute the entire pipeline end-to-end sequentially without modifying any paths.

### 1. Run the DSIGCN Graph Workflow

```bash
# Step 1: Parse raw mesh coordinates into text tokens
python Pre_mesh2graph/mesh_data_extractor.py

# Step 2: Build dual-scale graph objects (.bin)
python Pre_mesh2graph/Make_graph_bin.py

# Step 3: Run the DSIGCN GNN training pipeline
python train_aug.py

```

### 2. Run the 3D-CNN Baseline Workflow

```bash
# Step 1: Voxelize spatial sample matrices into output_mat
python 3DCNN_Baseline/Generate_mat.py

# Step 2: Train the 3D-CNN Model and save scaled predictions
python 3DCNN_Baseline/train_3dcnn.py

```

*This step trains a 3D Deep Convolutional network for 150 epochs, tracks validation loss checkpoints, performs automated min-max un-scaling, and outputs `test_set_predictions.csv`.*

### 3. Generate Benchmark Evaluation Plots

After training both models, run the plotting script to generate a publication-ready $3\times4$ error distribution violin grid comparing the structural limits of both networks:

```bash
python 3DCNN_Baseline/plot_vs.py

```

*The resulting high-DPI plot is saved to `example_data/Model_Comparison_Final/Error_Violins_SideBySide_Clean_3x4.png`.*

---

## Scaling to the Full Dataset

To train on your complete dataset, switch the relative example pointers to your absolute storage pathways:

1. **For DSIGCN (`train_aug.py`):**
```python
class Config:
    # Modify BIN_DIR to match your absolute full-dataset directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BIN_DIR = "/home/user/data/full_ellipsoid_dataset/" 

```


2. **For 3D-CNN (`3DCNN_Baseline/train_3dcnn.py`):**
```python
# Update your absolute cluster path arrays here
LABEL_CSV_PATH = "/absolute/path/to/full_dataset/ellipsoid_3d.csv"
PHASE_DATA_DIR = "/absolute/path/to/full_dataset/output_mat"
SPLIT_JSON_PATH = "/absolute/path/to/full_dataset/split_info.json"

```



---

## Citation

If you find this code or benchmark layout useful in your research, please consider citing our work:

> **[Insert Paper Title Here]** > *[Insert Author Names]* > Journal/Conference Name, Year.
> DOI: [Insert DOI Here]

```bibtex
@article{dsigcn2026,
  title={[Insert Paper Title Here]},
  author={[Insert Author Names]},
  journal={[Insert Journal Name]},
  year={2026},
  doi={[Insert DOI Here]}
}

```

```

```
