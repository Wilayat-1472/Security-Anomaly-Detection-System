import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Dict, Any, Optional, List
import io
import base64

from models.explainability import GradCAM


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str],
        confidence_threshold: float = 0.60,
        input_size: int = 224,
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize Grad-CAM if model exposes target layer
        self.grad_cam = None
        if hasattr(self.model, "get_target_layer_for_cam"):
            try:
                target_layer = self.model.get_target_layer_for_cam()
                self.grad_cam = GradCAM(self.model, target_layer)
            except Exception as e:
                print(f"Warning: Grad-CAM initialization failed: {e}")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        predicted_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "unknown"
        all_probs = {self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))}
        is_anomaly = (predicted_class != "normal")

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_anomaly": is_anomaly,
            "is_reliable": confidence >= self.confidence_threshold,
            "all_probabilities": all_probs,
            "class_index": pred_idx,
        }

    def explain(self, image: Image.Image, target_class: Optional[int] = None) -> Optional[Image.Image]:
        """
        Generates Grad-CAM visual attention overlay highlighting salient image regions.
        """
        if self.grad_cam is None:
            return None

        # Tensor must retain gradients for backward pass
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        heatmap = self.grad_cam.generate_heatmap(input_tensor, target_class=target_class)
        overlay = GradCAM.overlay_heatmap(image, heatmap, alpha=0.45, colormap="jet")
        return overlay

    def predict_with_visualization(self, image: Image.Image) -> Dict[str, Any]:
        result = self.predict(image)
        probs = result["all_probabilities"]
        result["probabilities_chart"] = {
            "labels": list(probs.keys()),
            "values": list(probs.values()),
        }

        # Generate explainability visual map
        cam_overlay = self.explain(image, target_class=result["class_index"])
        result["has_gradcam"] = cam_overlay is not None

        return result, cam_overlay
