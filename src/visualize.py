import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from src.models.temporal_vit import AdvancedTemporalViT

class ViTAttentionRollout:
    """
    Implements Attention Rollout / Grad-CAM extraction on the spatial ViT 
    to visualize spatial heatmaps.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, input_tensor):
        with torch.no_grad():
            # Pass input through ViT
            outputs = self.model.vit(pixel_values=input_tensor, output_attentions=True)
            self.attentions = outputs.attentions
            
        if not self.attentions:
            return np.zeros((input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3]))
            
        batch_size = self.attentions[0].shape[0]
        rollout = torch.eye(self.attentions[0].shape[-1]).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for attention in self.attentions:
            attention_heads_fused = attention.mean(dim=1) # Mean over heads
            attention_heads_fused += torch.eye(attention_heads_fused.shape[-1])
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            rollout = torch.bmm(attention_heads_fused, rollout)
            
        # CLS token spatial mapping
        mask = rollout[:, 0, 1:]
        
        spatial_dim = int(np.sqrt(mask.shape[-1]))
        mask = mask.view(batch_size, spatial_dim, spatial_dim).numpy()
        
        # Resize to original image size
        resized_masks = []
        for i in range(batch_size):
            m = cv2.resize(mask[i], (input_tensor.shape[3], input_tensor.shape[2]))
            m = m - np.min(m)
            m = m / (np.max(m) + 1e-8)
            resized_masks.append(m)
            
        return np.array(resized_masks)

def export_attention_visual_grid(frames_tensor, attention_masks, output_path):
    """
    Creates a visual grid of original frames and their attention heatmaps side-by-side.
    Complies with workspace rule: All Python computer vision pipelines must output visual grids.
    """
    seq_len = frames_tensor.shape[0]
    
    # Denormalize frames
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames_rgb = frames_tensor * std + mean
    frames_rgb = frames_rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0
    frames_rgb = frames_rgb.astype(np.uint8)
    
    fig, axes = plt.subplots(2, seq_len, figsize=(seq_len * 3, 6))
    if seq_len == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i in range(seq_len):
        frame = frames_rgb[i]
        heatmap = attention_masks[i]
        
        # Heatmap overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)
        
        axes[0, i].imshow(frame)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Frame {i+1}")
        
        axes[1, i].imshow(overlay)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Attention {i+1}")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visual grid to {output_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AdvancedTemporalViT(use_mocks=True).to(device)
    model.eval()
    
    dummy_frames = torch.randn(8, 3, 224, 224).to(device)
    
    rollout = ViTAttentionRollout(model)
    attention_masks = rollout(dummy_frames)
    
    export_path = "./figures/attention_grid.png"
    export_attention_visual_grid(dummy_frames.cpu(), attention_masks, export_path)
