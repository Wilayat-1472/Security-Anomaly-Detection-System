import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Deep Vision Surveillance Threat & Anomaly Detection Pipeline (Research & Production Grade)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational commands")

    # prepare
    prep_parser = subparsers.add_parser("prepare", help="Verify, split and structure dataset")
    prep_parser.add_argument("--data_dir", default="data", help="Raw dataset root folder")
    prep_parser.add_argument("--output_dir", default="data/processed", help="Target split dataset directory")
    prep_parser.add_argument("--train_split", type=float, default=0.8)
    prep_parser.add_argument("--val_split", type=float, default=0.1)

    # train
    train_parser = subparsers.add_parser("train", help="Train neural network architecture")
    train_parser.add_argument("--config", default="configs/config.yaml")
    train_parser.add_argument("--data_dir", default="data/processed")
    train_parser.add_argument("--model", type=str, default=None, help="Architecture: mobilenetv2, convnext_tiny, resnet50")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--batch_size", type=int)
    train_parser.add_argument("--loss", type=str, help="Loss: focal, weighted_ce, ce")
    train_parser.add_argument("--optimizer", type=str, help="Optimizer: adamw, sgd, adam")
    train_parser.add_argument("--no_pretrained", action="store_true")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Compute benchmark metrics and confusion matrix")
    eval_parser.add_argument("--config", default="configs/config.yaml")
    eval_parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    eval_parser.add_argument("--data_dir", default="data/processed")

    # predict
    pred_parser = subparsers.add_parser("predict", help="Execute inference on a target image with visual attribution")
    pred_parser.add_argument("image_path", help="Path to input image")
    pred_parser.add_argument("--config", default="configs/config.yaml")
    pred_parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    pred_parser.add_argument("--threshold", type=float, default=0.60)

    # web
    web_parser = subparsers.add_parser("web", help="Launch interactive explainable vision web dashboard")
    web_parser.add_argument("--config", default="configs/config.yaml")
    web_parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    web_parser.add_argument("--port", type=int, default=5000)
    web_parser.add_argument("--threshold", type=float)

    args = parser.parse_args()

    if args.command == "prepare":
        from data.prepare import split_dataset, verify_dataset
        print("Auditing raw dataset distribution:")
        verify_dataset(args.data_dir)
        print(f"\nStratifying dataset (Train: {args.train_split*100:.0f}%, Val: {args.val_split*100:.0f}%)...")
        split_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=1.0 - args.train_split - args.val_split,
        )
        print("\nProcessed dataset structure:")
        verify_dataset(args.output_dir)

    elif args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0], "--config", args.config, "--data_dir", args.data_dir]
        if args.model:
            sys.argv += ["--model", args.model]
        if args.epochs:
            sys.argv += ["--epochs", str(args.epochs)]
        if args.lr:
            sys.argv += ["--lr", str(args.lr)]
        if args.batch_size:
            sys.argv += ["--batch_size", str(args.batch_size)]
        if args.loss:
            sys.argv += ["--loss", args.loss]
        if args.optimizer:
            sys.argv += ["--optimizer", args.optimizer]
        if args.no_pretrained:
            sys.argv += ["--no_pretrained"]
        train_main()

    elif args.command == "evaluate":
        from scripts.evaluate import main as eval_main
        sys.argv = [sys.argv[0], "--config", args.config, "--checkpoint", args.checkpoint, "--data_dir", args.data_dir]
        eval_main()

    elif args.command == "predict":
        from scripts.predict import main as pred_main
        sys.argv = [sys.argv[0], args.image_path, "--config", args.config, "--checkpoint", args.checkpoint, "--threshold", str(args.threshold)]
        pred_main()

    elif args.command == "web":
        from scripts.run_webapp import main as web_main
        sys.argv = [sys.argv[0], "--config", args.config, "--checkpoint", args.checkpoint, "--port", str(args.port)]
        if args.threshold:
            sys.argv += ["--threshold", str(args.threshold)]
        web_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
