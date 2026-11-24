import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Launch Security Anomaly Detection Web Interface")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    args = parser.parse_args()

    config = load_config(args.config)

    from models.model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config["model"]["num_classes"] = len(config["classes"])
    model = create_model(config)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint}")
        print("The model will use untrained weights.")

    class_names = config["classes"]
    threshold = args.threshold if args.threshold is not None else config["inference"]["confidence_threshold"]

    from inference.webapp import start_app
    start_app(
        model=model,
        device=device,
        class_names=class_names,
        confidence_threshold=threshold,
        input_size=config["data"]["input_size"],
        port=args.port,
    )


if __name__ == "__main__":
    main()
