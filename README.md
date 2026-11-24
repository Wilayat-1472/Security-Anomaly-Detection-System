# Explainable Deep Vision Surveillance Anomaly Detection System (X-SADS)

An end-to-end, research-grade Computer Vision and Deep Learning framework designed for real-time automated security threat detection from CCTV / surveillance imagery. The system pairs high-efficiency vision backbones with gradient-weighted explainability (Grad-CAM), cost-sensitive Focal Loss, and an interactive diagnostic web telemetry console.

---

## 🔬 Key Architectural Highlights & Research Contributions

1. **Multi-Backbone Transfer Learning Suite:**
   - **MobileNetV2:** Designed for edge / low-power embedded surveillance hardware (~2.5M parameters).
   - **ConvNeXt-Tiny:** Modernized convolutional architecture incorporating Swin/ViT design principles for superior representation capacity.
   - **ResNet-50:** Benchmark baseline for comparative ablation studies.

2. **Explainable AI (XAI) with Grad-CAM:**
   - Integrates gradient-weighted class activation mapping (Selvaraju et al., ICCV 2017) directly into the inference engine.
   - Highlights the exact spatial regions triggering weapon, mask, smoke, or fire classifications, mitigating "black-box" decision vulnerabilities in mission-critical security tasks.

3. **Imbalance-Robust Optimization:**
   - **Focal Loss Formulation:** Penalizes easily classified background patterns to focus gradient updates on rare/hard threat examples.
   - **Class-Balanced Weighting:** Dynamic inverse-frequency weighting across target classes.
   - **Cosine Annealing with Warmup:** Adaptive learning rate scheduling preventing premature saddle-point stagnation.

4. **Rigorous Benchmark Evaluation Suite:**
   - Evaluates beyond raw accuracy: Macro & Weighted Precision, Recall, Specificity, False Positive Rate (FPR per class), and high-resolution Confusion Matrices.

5. **Diagnostic Web Telemetry Console:**
   - Asynchronous Flask/AJAX dashboard providing real-time tri-view telemetry: Original Frame, Grad-CAM Heatmap Attribution, and Diagnostic Anomaly Alerting.

---

## 📊 Classes & Threat Taxonomy

- `weapons`: Handguns, knives, tactical firearms
- `masks`: Gas masks, intruder concealments
- `fire`: Active open flame hazard
- `smoke`: Toxic particulate emissions and early fire indicators
- `normal`: Routine non-hazardous surveillance activity

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Recommended Python version: 3.10 - 3.12
pip install -r requirements.txt
```

### 2. Dataset Preparation
Organize training images by class in `data/<class_name>/` and stratify into `train` / `val` / `test`:
```bash
python run.py prepare --train_split 0.8 --val_split 0.1
```

### 3. Model Training
Train using AdamW, Cosine Annealing, and Focal Loss:
```bash
# MobileNetV2 default
python run.py train --epochs 35 --lr 0.0003 --loss focal --optimizer adamw

# Or train ConvNeXt-Tiny
python run.py train --model convnext_tiny --epochs 30 --lr 0.0001
```

### 4. Comprehensive Evaluation
Compute academic metrics and generate the confusion matrix:
```bash
python run.py evaluate --checkpoint checkpoints/best_model.pt
```

### 5. Single Image Prediction with Grad-CAM Attribution
```bash
python run.py predict "data/processed/test/weapons/Pasted image (3).png"
```

### 6. Interactive Telemetry Web Dashboard
```bash
python run.py web --port 5000
# Open http://localhost:5000 in your browser
```

---

## 📁 Repository Structure

```
├── configs/
│   └── config.yaml             # Central configuration (hyperparameters, models, augmentations)
├── data/
│   ├── prepare.py              # Stratification, class auditing & split verification
│   ├── dataloader.py           # Datasets, augmentations, and inverse class weights
│   ├── [class folders]         # Source surveillance images
│   └── processed/              # Stratified train/, val/, test/ partitions
├── models/
│   ├── model.py                # MobileNetV2, ConvNeXt-Tiny, ResNet-50 architectures
│   └── explainability.py       # Grad-CAM gradient backpropagation & overlay hooks
├── training/
│   └── trainer.py              # Focal Loss, Cosine LR scheduler, checkpointing
├── evaluation/
│   └── metrics.py              # Macro/Weighted P/R/F1, FPR, Confusion Matrix
├── inference/
│   ├── predict.py              # Inference engine with spatial explanation hooks
│   └── webapp.py               # Flask API serving predictions & heatmap overlays
├── templates/
│   └── index.html              # Modern dark-mode web console with tri-view inspection
├── static/                     # Persisted upload captures and evaluation charts
├── scripts/                    # Entrypoints for train, evaluate, predict, webapp
└── run.py                      # Master pipeline CLI
```
