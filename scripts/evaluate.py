import os
import sys
import argparse
import yaml
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained anomaly detection model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="data/processed", help="Path to processed dataset")
    args = parser.parse_args()

    config = load_config(args.config)

    from data.dataloader import create_dataloaders
    from models.model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config["data"]["batch_size"],
        input_size=config["data"]["input_size"],
        num_workers=config["data"]["num_workers"],
    )

    config["model"]["num_classes"] = len(class_names)
    model = create_model(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  Val accuracy in checkpoint: {checkpoint.get('val_acc', 'N/A')}")

    from evaluation.metrics import evaluate
    results = evaluate(model, test_loader, device, class_names, save_confusion_matrix=True)

    os.makedirs("static/results", exist_ok=True)
    results_serializable = {k: v for k, v in results.items() if k != "confusion_matrix"}
    with open("static/results/evaluation_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to static/results/evaluation_results.json")


if __name__ == "__main__":
    main()
