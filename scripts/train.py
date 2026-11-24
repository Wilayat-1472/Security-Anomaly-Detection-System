import os
import sys
import argparse
import yaml
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import create_dataloaders
from models.model import create_model
from training.trainer import Trainer
from evaluation.metrics import evaluate


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Security Anomaly Detection Model with Research Capabilities")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", default="data/processed", help="Path to split dataset")
    parser.add_argument("--model", type=str, default=None, help="Model architecture: mobilenetv2, convnext_tiny, resnet50")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--loss", type=str, default=None, help="Loss function: focal, weighted_ce, ce")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer: adamw, sgd, adam")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable pretrained weights")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.loss:
        config["training"]["loss_type"] = args.loss
    if args.optimizer:
        config["training"]["optimizer"] = args.optimizer
    if args.no_pretrained:
        config["model"]["pretrained"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")

    # Create dataloaders with class frequency weighting
    train_loader, val_loader, test_loader, class_names, class_weights = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config["data"]["batch_size"],
        input_size=config["data"]["input_size"],
        num_workers=config["data"]["num_workers"],
    )
    print(f"Registered Classes ({len(class_names)}): {class_names}")
    print(f"Batch distribution -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    print(f"Class-balancing weights: {class_weights.numpy().round(3)}")

    config["model"]["num_classes"] = len(class_names)
    model = create_model(config)

    trainer = Trainer(
        model=model,
        device=device,
        config=config,
        class_names=class_names,
        class_weights=class_weights,
    )
    history = trainer.fit(train_loader, val_loader)

    # Load optimal checkpoint and evaluate on untouched test distribution
    trainer.load_best()
    results = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_confusion_matrix=True,
    )

    # Persist evaluation metrics for tracking and academic reporting
    os.makedirs("static/results", exist_ok=True)
    results_serializable = {k: v for k, v in results.items() if k != "confusion_matrix"}
    with open("static/results/evaluation_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Evaluation benchmark successfully stored at static/results/evaluation_results.json")


if __name__ == "__main__":
    main()
