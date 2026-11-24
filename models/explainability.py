import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Generates visual explanations for CNN decisions by tracking gradients
    flowing into the final convolutional feature maps.
    Ref: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computes 2D normalized Grad-CAM heatmap array [0, 1] for input_tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        loss = output[0, target_class]
        loss.backward(retain_graph=True)

        # Global average pooling of gradients: weights = mean(gradients, spatial_dims)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain features that positively contribute to target class
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Blends heatmap over original PIL image.
        """
        img_np = np.array(image.convert("RGB"))
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        heatmap_resized_np = np.array(heatmap_resized) / 255.0

        cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(heatmap_resized_np)[:, :, :3]  # drop alpha
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

        blended = (1.0 - alpha) * img_np + alpha * colored_heatmap
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)
