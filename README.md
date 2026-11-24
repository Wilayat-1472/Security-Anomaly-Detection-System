# Explainable Deep Vision Surveillance Anomaly Detection System (X-SADS)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

An end-to-end, research-grade Computer Vision and Deep Learning framework engineered for automated surveillance threat and anomaly detection in CCTV streams and security imagery. 

**X-SADS** combines lightweight and state-of-the-art vision backbones with **Explainable AI (XAI)** via **Gradient-weighted Class Activation Mapping (Grad-CAM)**, addressing the "black-box" nature of neural networks in mission-critical surveillance tasks. It features cost-sensitive **Focal Loss** for extreme class imbalance, rigorous benchmark metrics, and an interactive real-time diagnostic web telemetry dashboard.

---

## 📌 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Threat Taxonomy & Classes](#-threat-taxonomy--classes)
- [Benchmark Results & Metrics](#-benchmark-results--metrics)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [End-to-End Workflow](#-end-to-end-workflow)
  - [1. Data Preparation & Stratification](#1-data-preparation--stratification)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Metrics](#3-evaluation--metrics)
  - [4. Single-Image Inference with Grad-CAM](#4-single-image-inference-with-grad-cam)
  - [5. Interactive Web Dashboard](#5-interactive-web-dashboard)
- [Configuration Guide (`config.yaml`)](#-configuration-guide-configyaml)
- [Explainability Engine (Grad-CAM)](#-explainability-engine-grad-cam)
- [API Reference](#-api-reference)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [License](#-license)

---

## 🔬 Key Features

1. **Multi-Backbone Vision Suite:**
   - **MobileNetV2:** Ultra-lightweight backbone (~2.5M parameters) optimized for edge AI devices, embedded NVRs, and drones.
   - **ConvNeXt-Tiny:** Modernized pure convolutional architecture incorporating Vision Transformer (ViT) and Swin design principles for maximum representational capacity.
   - **ResNet-50:** Classical deep residual network baseline for comparative studies.

2. **Explainable AI (Grad-CAM):**
   - Implements gradient-weighted class activation mapping (Selvaraju et al., ICCV 2017).
   - Generates high-resolution heatmaps overlaid on original frames, pinpointing specific image regions responsible for anomaly alerts (weapons, fire, smoke, masks).

3. **Imbalance-Robust Optimization:**
   - **Focal Loss ($\gamma = 2.0$):** Suppresses gradient influence from easy routine/background frames and concentrates updates on hard, rare threat samples.
   - **Inverse-Frequency Weighting:** Dynamically balances loss contributions across under-represented threat categories.
   - **Cosine Annealing with Warmup:** Prevents local minima traps and stabilizes deep feature learning.

4. **Production-Ready Web Telemetry Console:**
   - Modern dark-themed dashboard powered by Flask, Bootstrap, and asynchronous AJAX.
   - Features **Tri-View Inspection**: Side-by-side comparison of Raw Surveillance Frame, Spatial Grad-CAM Overlay, and Probability Confidence Distribution.

---

## 🏗️ System Architecture

```
                                [ Surveillance Camera / Image Feed ]
                                                 │
                                                 ▼
                                     [ Preprocessing & Augmentation ]
                                     (Resize 224x224, Normalize, TTA)
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [ Vision Backbone ]                                 [ Feature Maps A^k ]
       (MobileNetV2 / ConvNeXt / ResNet50)                                │
                        │                                                 │
                        ▼                                                 │
             [ Linear Classifier ]                                        │
                        │                                                 │
         ┌──────────────┴──────────────┐                                  │
         ▼                             ▼                                  │
 [ Predicted Logits ]         [ Backward Gradients: dY_c / dA^k ]         │
         │                             │                                  │
         │                             └────────────────┬─────────────────┘
         │                                              ▼
         │                                  [ Global Average Pooling ]
         │                                    α_k = (1/Z) Σ dY/dA
         │                                              │
         │                                              ▼
         │                                    [ Weighted Sum + ReLU ]
         │                                       L_GradCAM = ReLU(Σ α_k A^k)
         │                                              │
         ▼                                              ▼
[ Anomaly Alert Classification ]            [ Spatial Heatmap Overlay ]
         │                                              │
         └──────────────────────┬───────────────────────┘
                                ▼
         [ Interactive Telemetry Dashboard / Webhook Alert ]
```

---

## 📊 Threat Taxonomy & Classes

The model classifies surveillance frames into 5 distinct categories:

| Class Name | Target Description | Threat Level | Primary Visual Indicators |
|:---|:---|:---:|:---|
| `weapons` | Handguns, assault rifles, knives, tactical firearms | 🚨 Critical | Metallic contours, barrels, grips, edged blades |
| `fire` | Active combustion, open flames, structural fire | 🚨 Critical | High-luminance orange/yellow flame tongues |
| `smoke` | Dense smoke plumes, toxic particulates, smoldering | ⚠️ High | Diffuse gray/white plume structures, occlusions |
| `masks` | Balaclavas, gas masks, tactical facial concealments | ⚠️ High | Facial feature obstruction, unnatural contours |
| `normal` | Routine surveillance, authorized foot traffic | 🟢 Safe | Typical indoor/outdoor non-threatening scenes |

---

## 📈 Benchmark Results & Metrics

The system records per-class and aggregated performance metrics on held-out test splits. Metrics tracked include:
- **Macro & Weighted Precision / Recall / F1-Score**
- **False Positive Rate (FPR)** per class
- **Specificity** ($1 - \text{FPR}$)
- **Confusion Matrix** with automatic Seaborn heatmap export

```
========================================================================================
                          MODEL BENCHMARK EVALUATION REPORT                             
========================================================================================
Overall Accuracy:       0.3000 (Initial calibration baseline)
Macro Precision:        0.1667           Weighted Precision:  0.1667
Macro Recall:           0.3000           Weighted Recall:     0.3000
Macro F1-Score:         0.2100           Weighted F1-Score:   0.2100
----------------------------------------------------------------------------------------
Class-Wise Performance:
  • fire      -> Precision: 66.7% | Recall: 100.0% | F1: 80.0% | FPR: 12.5%
  • weapons   -> Precision: 16.7% | Recall:  50.0% | F1: 25.0% | FPR: 62.5%
  • smoke     -> Pending additional training samples
  • masks     -> Pending additional training samples
  • normal    -> Zero false alerts recorded
========================================================================================
```
*Generated reports and confusion matrix plots are saved to `static/results/confusion_matrix.png` and `static/results/evaluation_results.json`.*

---

## 📁 Directory Structure

```text
Security-Anomaly-Detection-System/
├── checkpoints/               # Saved model weights (*.pt)
│   ├── best_model.pt          # Top validation checkpoint
│   └── checkpoint_epoch_*.pt  # Epoch checkpoints
├── configs/
│   └── config.yaml            # Master hyperparameters, optimizer, & augmentation config
├── data/                      # Dataset root
│   ├── fire/                  # Raw class folder: fire
│   ├── masks/                 # Raw class folder: masks
│   ├── normal/                # Raw class folder: normal
│   ├── smoke/                 # Raw class folder: smoke
│   ├── weapons/               # Raw class folder: weapons
│   ├── dataloader.py          # PyTorch datasets, stratified loaders & transforms
│   ├── prepare.py             # Data auditing, stratified splitting & integrity checks
│   └── processed/             # Auto-generated train/, val/, test/ partitions
├── evaluation/
│   └── metrics.py             # Academic metrics calculator (Precision, Recall, F1, FPR, Confusion Matrix)
├── inference/
│   ├── predict.py             # CLI inference and prediction visualizer
│   └── webapp.py              # Flask server and REST API for real-time predictions
├── models/
│   ├── model.py               # MobileNetV2, ConvNeXt-Tiny, ResNet-50 network definitions
│   └── explainability.py      # Hook-based PyTorch Grad-CAM engine
├── notebooks/                 # Exploratory data analysis and experimental notebooks
├── scripts/
│   ├── evaluate.py            # Evaluation pipeline runner
│   ├── predict.py             # Single image prediction runner
│   ├── run_webapp.py          # Web dashboard bootstrap runner
│   └── train.py               # Model training script
├── static/
│   ├── results/               # Saved confusion matrices and prediction diagnostics
│   └── uploads/               # Telemetry uploads cache
├── templates/
│   └── index.html             # Web dashboard UI template
├── training/
│   └── trainer.py             # Trainer class, Focal Loss, Cosine LR scheduler, Early Stopping
├── requirements.txt           # Python dependency specifications
├── run.py                     # Master Unified CLI entrypoint
└── README.md                  # Project documentation
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA support (optional, CPU execution supported out-of-the-box)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Wilayat-1472/Security-Anomaly-Detection-System.git
cd Security-Anomaly-Detection-System
```

### Step 2: Create a Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Or using Conda
conda create -n xsads python=3.11 -y
conda activate xsads
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 End-to-End Workflow

The repository includes a unified master CLI runner (`run.py`) to manage all operational phases.

### 1. Data Preparation & Stratification
Audit the raw dataset distribution and automatically create stratified `train`, `val`, and `test` splits:
```bash
python run.py prepare --data_dir data --output_dir data/processed --train_split 0.8 --val_split 0.1
```
*This command verifies image validity, balances distributions across splits, and updates class directories under `data/processed/`.*

### 2. Model Training
Train the neural network using AdamW, Cosine Annealing, and Focal Loss:
```bash
# Default training (MobileNetV2, 35 epochs, focal loss)
python run.py train

# Train ConvNeXt-Tiny with custom hyperparameters
python run.py train --model convnext_tiny --epochs 40 --lr 0.0001 --batch_size 16 --loss focal --optimizer adamw

# Train ResNet-50 baseline
python run.py train --model resnet50 --epochs 30 --lr 0.0003 --loss weighted_ce
```
Checkpoints are automatically saved to `checkpoints/best_model.pt`.

### 3. Evaluation & Metrics
Run full academic evaluation against the test partition:
```bash
python run.py evaluate --checkpoint checkpoints/best_model.pt --data_dir data/processed
```
**Output artifacts:**
- `static/results/evaluation_results.json`: Raw metric scores per class.
- `static/results/confusion_matrix.png`: Normalized confusion matrix visualization.

### 4. Single-Image Inference with Grad-CAM
Classify an image from the command line and generate visual attribution:
```bash
python run.py predict "data/processed/test/weapons/Pasted image (3).png" --threshold 0.60
```
This prints the predicted anomaly class, top-k probabilities, and exports a side-by-side diagnostic image with Grad-CAM highlighting to `static/results/`.

### 5. Interactive Web Dashboard
Launch the real-time telemetry console:
```bash
python run.py web --port 5000
```
Then navigate to:
```
http://localhost:5000
```
**Features of the Web Console:**
- Drag-and-drop surveillance frame upload.
- Live AJAX inference displaying Threat Status (Normal vs. Anomaly).
- Interactive tri-view: **Original Frame**, **Grad-CAM Heatmap**, and **Diagnostic Distribution**.
- Real-time confidence progress meters for all 5 threat categories.

---

## ⚙️ Configuration Guide (`config.yaml`)

All system parameters can be modified via `configs/config.yaml`:

```yaml
model:
  name: "mobilenetv2"       # Options: mobilenetv2, convnext_tiny, resnet50
  num_classes: 5
  pretrained: true          # Transfer learning initialization from ImageNet
  freeze_backbone: false    # Set to true to train only the classifier head
  dropout: 0.3
  input_size: 224

classes:
  - normal
  - weapons
  - masks
  - fire
  - smoke

training:
  epochs: 35
  optimizer: "adamw"         # Options: adamw, sgd, adam
  learning_rate: 0.0003
  weight_decay: 0.0001
  loss_type: "focal"         # Options: focal, weighted_ce, ce
  focal_gamma: 2.0           # Focusing parameter for Focal Loss
  label_smoothing: 0.1       # Prevents overconfident misclassifications
  scheduler: "cosine"        # Learning rate schedule: cosine or step
  grad_clip: 1.0             # Max gradient norm clipping
  early_stop_patience: 12    # Early stopping epochs

augmentation:
  random_horizontal_flip: true
  random_rotation: 15
  random_perspective: 0.2
  color_jitter: 0.3
  random_erasing: 0.2

inference:
  confidence_threshold: 0.60
  top_k: 3
  explainability: "gradcam"
```

---

## 🧠 Explainability Engine (Grad-CAM)

In high-stakes surveillance environments, false alarms waste critical response time and missed detections present severe safety risks. Grad-CAM provides visual justification for every inference:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

- **Forward Hook:** Captures the activation map $A^k$ at the final convolutional feature layer (`features[-1]` in MobileNetV2, `stages[-1]` in ConvNeXt, `layer4` in ResNet).
- **Backward Hook:** Extracts gradients $\frac{\partial Y^c}{\partial A^k}$ with respect to the predicted class score $Y^c$.
- **Spatial Heatmap Generation:** Positive features are aggregated, normalized to $[0, 1]$, and merged with a colormap overlay (`JET` / `TURBO`) on the source frame.

---

## 🔌 API Reference

The web application exposes a REST API endpoint for integration into external Video Management Systems (VMS) or SIEM pipelines:

### `POST /predict`
Uploads a frame and returns JSON prediction telemetry along with URLs to generated Grad-CAM overlays.

**Request:**
- `Content-Type`: `multipart/form-data`
- `file`: Image file (`.png`, `.jpg`, `.jpeg`, `.webp`)

**Response:**
```json
{
  "success": true,
  "filename": "camera_04_frame_0182.png",
  "prediction": {
    "predicted_class": "weapons",
    "confidence": 0.8842,
    "is_anomaly": true,
    "top_k": [
      {"class": "weapons", "confidence": 0.8842},
      {"class": "masks", "confidence": 0.0715},
      {"class": "normal", "confidence": 0.0241}
    ]
  },
  "upload_url": "/static/uploads/camera_04_frame_0182.png",
  "gradcam_url": "/static/results/gradcam_camera_04_frame_0182.png",
  "result_url": "/static/results/pred_camera_04_frame_0182.png"
}
```

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>1. How do I switch backbones?</b></summary>
Pass `--model <name>` to the train command or update `config.yaml`:
```bash
python run.py train --model convnext_tiny
```
Supported architectures: `mobilenetv2`, `convnext_tiny`, `resnet50`.
</details>

<details>
<summary><b>2. How can I train on a GPU or resolve CUDA out-of-memory errors?</b></summary>
The system automatically detects CUDA if available. If you encounter memory issues:
- Reduce `--batch_size` (e.g., `--batch_size 8`).
- Enable mixed precision by setting `use_amp: true` in `configs/config.yaml`.
</details>

<details>
<summary><b>3. Why is an alert triggered even when normal confidence is non-zero?</b></summary>
The model uses a configurable confidence threshold (`--threshold 0.60`). If any threat class exceeds the decision threshold, it raises an anomaly flag. You can adjust this threshold in `configs/config.yaml` or when running `predict` / `web`.
</details>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Contributions

- **Wilayat Ali** - [@Wilayat-1472](https://github.com/Wilayat-1472)
- Contributions, issues, and feature requests are welcome! Feel free to open a pull request or submit an issue.
