import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import base64
from PIL import Image

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    for visual explanations of CNN decisions.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else getattr(model, 'layer4', None)
        
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        if self.target_layer is not None:
            h1 = self.target_layer.register_forward_hook(forward_hook)
            h2 = self.target_layer.register_full_backward_hook(backward_hook)
            self.hook_handles.extend([h1, h2])

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate raw Grad-CAM activation heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
            
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            # Fallback if layer hooks weren't triggered
            cam = np.zeros((224, 224), dtype=np.float32)
            return cam
            
        gradients = self.gradients.detach().cpu().data.numpy()[0]
        activations = self.activations.detach().cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of feature maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # ReLU to keep only features that have a positive influence on the class
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam_max = np.max(cam)
        cam_min = np.min(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        cam = cv2.resize(cam, (224, 224))
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def generate_gradcam_overlays(model, input_tensor, original_pil_image, target_class=None, alpha=0.5):
    """
    Generate base64 encoded original, heatmap, and blended overlay images.
    """
    grad_cam = GradCAM(model, target_layer=getattr(model, 'layer4', getattr(getattr(model, 'backbone', None), 'layer4', None)))
    cam = grad_cam.generate_heatmap(input_tensor, target_class=target_class)
    grad_cam.remove_hooks()
    
    # Resize original image to 224x224
    orig_np = np.array(original_pil_image.resize((224, 224)).convert('RGB'))
    
    # Colormap heatmap
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blended Overlay
    overlay = np.uint8(alpha * heatmap_rgb + (1 - alpha) * orig_np)
    
    # Base64 encodings
    def to_base64(img_arr):
        pil_img = Image.fromarray(img_arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        
    return {
        "original": to_base64(orig_np),
        "heatmap": to_base64(heatmap_rgb),
        "overlay": to_base64(overlay)
    }
