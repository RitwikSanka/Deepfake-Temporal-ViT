import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from src.models.temporal_vit import AdvancedTemporalViT, TemporalViTBaseline
from transformers import ViTModel, ViTConfig
from torch.utils.data import Dataset, DataLoader
from src.visualize import ViTAttentionRollout

class StaticViTBaseline(nn.Module):
    def __init__(self, hidden_size=768, use_mocks=False):
        super().__init__()
        if not use_mocks:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', attn_implementation="eager")
        else:
            config = ViTConfig(hidden_size=hidden_size, num_hidden_layers=2, num_attention_heads=4)
            config._attn_implementation = "eager"
            self.vit = ViTModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x, residuals=None):
        batch, seq, c, h, w = x.shape
        x_flat = x.view(batch*seq, c, h, w)
        out = self.vit(pixel_values=x_flat).last_hidden_state[:, 0, :]
        return self.classifier(out)

class CalibratedSyntheticDataset(Dataset):
    """
    Lightweight synthetic calibration routine: generates face crops with subtle, 
    structured temporal artifacts (high-frequency residual noise along the blending boundaries 
    for deepfakes vs. smooth optical flow for reals).
    """
    def __init__(self, num_samples=100, seq_len=8, image_size=224):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.image_size = image_size
        # Balanced dataset
        self.labels = np.array([1 if i < num_samples // 2 else 0 for i in range(num_samples)])
        np.random.shuffle(self.labels)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        x = torch.zeros(self.seq_len, 3, self.image_size, self.image_size)
        
        base_color = torch.tensor([0.8, 0.6, 0.5]).view(3, 1, 1)
        Y, X = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
        
        noise_level = 0.08
        jitter_prob = 0.5
        
        for t in range(self.seq_len):
            frame = base_color.clone().repeat(1, self.image_size, self.image_size)
            
            # Smooth optical flow for real
            center_x = self.image_size // 2 + int(10 * np.sin(t * 0.5))
            center_y = self.image_size // 2 + int(10 * np.cos(t * 0.5))
            
            # Temporal discontinuity / micro-jitters for fakes
            if label == 1 and np.random.rand() < jitter_prob:
                center_x += np.random.randint(-4, 5)
                center_y += np.random.randint(-4, 5)
                
            dist = (X - center_x)**2 + (Y - center_y)**2
            mask = dist < (50**2)
            frame[:, mask] = torch.tensor([0.9, 0.7, 0.6]).view(3, 1)
            
            # High-frequency residual noise along the blending boundaries for deepfakes
            if label == 1 and np.random.rand() < 0.6:
                boundary_mask = (dist > (45**2)) & (dist < (55**2))
                noise = torch.randn(3, self.image_size, self.image_size) * noise_level
                frame[:, boundary_mask] += noise[:, boundary_mask]
                
            x[t] = frame.clamp(0, 1)
            
        residuals = []
        for i in range(self.seq_len):
            if i == 0:
                residuals.append(torch.zeros_like(x[0]))
            else:
                residuals.append(torch.abs(x[i] - x[i-1]))
        residuals = torch.stack(residuals)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        return x, residuals, torch.tensor(label, dtype=torch.float32)

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx]

def measure_fps(model, device, seq_len=8, batch_size=1, num_trials=10):
    model.to(device)
    model.eval()
    x = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
    res = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
    for _ in range(3):
        with torch.no_grad():
            model(x, res)
    latencies = []
    for _ in range(num_trials):
        start = time.time()
        with torch.no_grad():
            model(x, res)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        latencies.append(end - start)
    latencies = np.array(latencies)
    fps = (seq_len * batch_size) / latencies
    return np.mean(fps), np.std(fps)

def train_model(model, dataloader, device, is_static=False, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        for x, res, label in dataloader:
            x, res, label = x.to(device), res.to(device), label.to(device)
            if is_static:
                x, res = x[:, :1], res[:, :1]
            optimizer.zero_grad()
            prob = model(x, res).squeeze(-1)
            loss = criterion(prob, label)
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataloader, device, is_static=False):
    model.to(device)
    model.eval()
    
    y_true, y_scores = [], []
    with torch.no_grad():
        for x, res, label in dataloader:
            x, res = x.to(device), res.to(device)
            if is_static:
                x, res = x[:, :1], res[:, :1]
            prob = model(x, res).cpu().numpy().flatten()
            y_scores.extend(prob)
            y_true.extend(label.cpu().numpy().flatten())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        eer = calculate_eer(y_true, y_scores)
    else:
        auc = 0.5
        eer = 0.5
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return y_true, y_scores, y_pred, auc, eer, acc, f1

def run_benchmark():
    os.makedirs("/home/doom/deepfake_detection/figures", exist_ok=True)
    
    print("Generating calibrated structured temporal dataset...")
    train_dataset = CalibratedSyntheticDataset(num_samples=10, seq_len=8)
    test_dataset = CalibratedSyntheticDataset(num_samples=20, seq_len=8)
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    models = {
        "Static ViT": StaticViTBaseline(use_mocks=False),
        "ViT + GRU": TemporalViTBaseline(use_mocks=False),
        "Advanced Temporal ViT": AdvancedTemporalViT(use_mocks=False)
    }
    
    results = []
    roc_data, cm_data = {}, {}
    device_cpu = "cpu"
    device_gpu = "cuda" if torch.cuda.is_available() else None
    
    for name, model in models.items():
        print(f"\nTraining {name} for 10 epochs on calibrated set...")
        is_static = (name == "Static ViT")
        seq_len_fps = 1 if is_static else 8
        
        train_model(model, train_loader, device_cpu, is_static=is_static, epochs=10)
        
        print(f"Evaluating {name}...")
        fps_cpu_mean, fps_cpu_std = measure_fps(model, device_cpu, seq_len=seq_len_fps)
        
        y_true, y_scores, y_pred, auc, eer, acc, f1 = evaluate_model(model, test_loader, device_cpu, is_static=is_static)
        
        roc_data[name] = (y_true, y_scores)
        cm_data[name] = confusion_matrix(y_true, y_pred, normalize='true')
        
        res_dict = {
            "Model": name,
            "AUC-ROC": f"{auc:.4f}",
            "EER": f"{eer:.4f}",
            "Accuracy": f"{acc:.4f}",
            "F1-Score": f"{f1:.4f}",
            "FPS (CPU) Mean": f"{fps_cpu_mean:.2f}",
            "FPS (CPU) Std": f"{fps_cpu_std:.2f}",
            "FPS (GPU) Mean": "N/A", "FPS (GPU) Std": "N/A"
        }
        results.append(res_dict)
        
    df = pd.DataFrame(results)
    df.to_csv("./figures/benchmark_table.csv", index=False)
    print("\nBenchmark Results:")
    print(df.to_string())
    
    plt.figure(figsize=(8, 6))
    for name, (yt, ys) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, ys)
        plt.plot(fpr, tpr, label=f"{name}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Calibrated Evaluation)')
    plt.legend()
    plt.savefig("./figures/roc_curve.png")
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax.set_title(f"{name}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig("./figures/confusion_matrix.png")
    plt.close()
    
    print("\nGenerating Attention Maps...")
    adv_model = models["Advanced Temporal ViT"]
    adv_model.eval().to('cpu')
    
    x, res, label = None, None, None
    for i in range(len(test_dataset)):
        if test_dataset[i][2] == 1:
            x, res, label = test_dataset[i]
            break
            
    x_sample = x[:4]
    res_sample = res[:4]
    rollout = ViTAttentionRollout(adv_model)
    dummy_input = x_sample.unsqueeze(0)
    masks = rollout(dummy_input.view(4, 3, 224, 224))
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(4):
        frame = x_sample[i] * std + mean
        frame = frame.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, i].imshow(frame)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Aligned Face (Fake)", fontsize=10)
        
        residual = res_sample[i].permute(1, 2, 0).numpy()
        res_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        axes[1, i].imshow(res_norm)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Residual Motion", fontsize=10)
        
        heatmap = masks[i]
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * frame + 0.5 * heatmap_colored
        axes[2, i].imshow(overlay)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Attention Heatmap", fontsize=10)
        
    plt.tight_layout()
    plt.savefig("./figures/attention_maps.png")
    plt.close()
    
    print("Benchmark complete. Results saved to figures/.")

if __name__ == "__main__":
    run_benchmark()
