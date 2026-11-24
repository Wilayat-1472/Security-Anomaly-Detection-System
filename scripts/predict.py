import os
import sys
import argparse
import yaml
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Predict anomaly in a single image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold")
    args = parser.parse_args()

    config = load_config(args.config)

    from models.model import create_model
    from inference.predict import Predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config["model"]["num_classes"] = len(config["classes"])
    model = create_model(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    class_names = checkpoint.get("class_names", config["classes"])
    predictor = Predictor(model, device, class_names, confidence_threshold=args.threshold)

    image = Image.open(args.image_path).convert("RGB")
    result = predictor.predict(image)

    print(f"\n=== Detection Result ===")
    print(f"Image: {args.image_path}")
    print(f"Prediction: {result['predicted_class'].upper()}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"Threat: {'⚠ ANOMALY DETECTED' if result['is_anomaly'] else '✓ SAFE'}")
    print(f"Reliable: {'Yes' if result['is_reliable'] else 'No — manual review needed'}")
    print(f"\nAll probabilities:")
    for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {cls:15s} | {bar:40s} | {prob*100:5.1f}%")


if __name__ == "__main__":
    main()
