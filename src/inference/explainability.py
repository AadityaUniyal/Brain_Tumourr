"""
High-Resolution Grad-CAM (Gradient-Weighted Class Activation Mapping):
Provides verifiable visual explainability for medical imaging.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Computes class activation maps for convolutional neural networks.
    Hooks into the final convolutional feature extractor.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        # Target layer must be in the feature pathway leading to classification logits
        self.target_layer = target_layer or model.encoder_stage4
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        input_tensor: [1, 3, H, W]
        target_class: int or None (defaults to argmax class)
        Returns:
            heatmap (np.ndarray): [H, W] normalized in [0, 1]
            predicted_class (int)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        outputs = self.model(input_tensor)
        logits = outputs["logits"]

        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        self.model.zero_grad()
        loss = logits[0, target_class]
        loss.backward(retain_graph=True)

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        activations = self.activations[0].clone()
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU on weighted map

        # Resize to input resolution
        _, _, h, w = input_tensor.shape
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 1e-7:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap, target_class

    def overlay_on_image(self, original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Superimposes the Grad-CAM heatmap over the original grayscale/RGB MRI scan.
        original_image: np.ndarray [H, W] or [H, W, 3] in [0, 1] or [0, 255]
        """
        if original_image.dtype != np.uint8:
            if np.max(original_image) <= 1.0:
                base_img = (original_image * 255).astype(np.uint8)
            else:
                base_img = original_image.astype(np.uint8)
        else:
            base_img = original_image

        if len(base_img.shape) == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        overlay = cv2.addWeighted(base_img, 1.0 - alpha, colored_heatmap, alpha, 0)
        return overlay
