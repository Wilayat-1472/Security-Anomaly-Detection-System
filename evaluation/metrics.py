import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    save_confusion_matrix: bool = True,
    cm_output_path: str = "static/results/confusion_matrix.png",
) -> Dict[str, Any]:
    """
    Computes rigorous academic benchmark evaluation metrics:
    - Global Accuracy, Macro & Weighted Precision, Recall, F1
    - Per-class True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    - Specificity / False Positive Rate (critical for surveillance alarm precision)
    - Formatted Confusion Matrix plot
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
    }

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    results["confusion_matrix"] = cm.tolist()

    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "specificity": specificity,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    results["per_class"] = per_class

    if save_confusion_matrix:
        _plot_confusion_matrix(cm, class_names, cm_output_path)

    print("\n" + "=" * 50)
    print("      COMPREHENSIVE EVALUATION BENCHMARK")
    print("=" * 50)
    print(f"Overall Accuracy:        {results['accuracy']:.4f}")
    print(f"Macro Precision:         {results['precision_macro']:.4f}")
    print(f"Macro Recall:            {results['recall_macro']:.4f}")
    print(f"Macro F1-Score:          {results['f1_macro']:.4f}")
    print(f"Weighted F1-Score:       {results['f1_weighted']:.4f}")
    print("\nPer-Class Breakdown:")
    for name, m in per_class.items():
        print(f"  [{name.upper():<8}] P={m['precision']:.3f} | R={m['recall']:.3f} | F1={m['f1']:.3f} | FPR={m['fpr']:.3f} (FP={m['fp']}, FN={m['fn']})")
    print("=" * 50 + "\n")

    return results


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=True,
    )
    plt.title("Confusion Matrix - Surveillance Anomaly Classification", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
