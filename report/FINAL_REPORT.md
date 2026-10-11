# Real-Time Deepfake Detection via Dual-Stream Temporal Vision Transformer

**Institution:** Department of Electrical Engineering, Indian Institute of Technology Madras (IIT Madras)  
**Contact Email:** [ee23b149@smail.iitm.ac.in](mailto:ee23b149@smail.iitm.ac.in)

---

## Executive Summary
This project implements an end-to-end, dual-stream deep learning architecture designed for real-time deepfake video detection. Unlike conventional frame-based detectors that focus solely on static spatial artifacts, our system simultaneously models high-level spatial facial semantics via a Vision Transformer (ViT-B/16) and micro-expression temporal discrepancies via an optical residual difference stream, cross-attention fusion, and a bidirectional GRU.

Evaluated on the benchmark **Celeb-DF v2** dataset, the system achieves **0.8834 AUC-ROC**, **82.00% Accuracy**, and **86.54% Deepfake Recall**, operating at **45.04 FPS on GPU** (surpassing real-time video requirements).

---

## 1. Key Performance & Evaluation Results (Celeb-DF v2)

| Metric | Real Empirical Value | Forensic Significance |
| :--- | :---: | :--- |
| **ROC AUC** | **0.8834** | High discriminative power across all decision thresholds. |
| **Accuracy** | **82.00%** | 164 of 200 unseen validation videos correctly classified. |
| **Deepfake Recall** | **86.54%** | Successfully flags 90 out of 104 synthesized video clips. |
| **Real Specificity** | **77.08%** | Correctly identifies 74 out of 96 authentic video clips. |
| **F1-Score** | **0.8334** | Harmonic mean reflecting balanced precision and recall. |
| **EER (Equal Error Rate)** | **0.1620** | Operating point where false accepts equal false rejects. |

---

## 2. Model Architecture & Efficiency Profile

### Parameter Breakdown
| Module | Parameters | % of Total | Primary Role |
| :--- | ---: | ---: | :--- |
| **Spatial ViT-B/16 Backbone** | 86,389,248 | 86.83% | Global semantic facial representations via `[CLS]` token |
| **Residual CNN Stream** | 69,312 | 0.07% | High-frequency inter-frame motion residuals |
| **Cross-Stream Linear Fusion** | 1,180,416 | 1.19% | Projects concatenated tokens (1536 $\rightarrow$ 768) |
| **Multi-Head Temporal Attention** | 11,027,968 | 11.08% | Models inter-frame temporal dependencies |
| **Bidirectional GRU** | 787,968 | 0.79% | Temporal sequence aggregation |
| **Classification MLP Head** | 33,025 | 0.03% | Binary prediction logit / probability |
| **Total Model Parameters** | **99,487,937** | **100.0%** | **Full network parameter capacity** |
| **Trainable (Fine-tuning)** | **20,186,561** | **20.29%** | **Active gradient memory during training** |

### Computational Complexity & FLOPs
* **FLOPs per Frame:** 17.67 GFLOPs (8.84 GMACs)
* **FLOPs per 8-Frame Sequence:** 141.37 GFLOPs (70.68 GMACs)
* **Memory Footprint:** 379.52 MB (FP32 single precision) / 189.76 MB (FP16 mixed precision)

---

## 3. Inference Throughput & Latency

* **CPU Throughput (Local Machine):**
  * Sequence Latency (8 frames): $1275.85 \pm 178.4$ ms
  * Per-Frame Latency: $159.48$ ms
  * Processing Speed: **6.27 FPS** (suitable for background forensic audit)
* **GPU Throughput (Nvidia T4, FP16 Autocast):**
  * Sequence Latency (8 frames): $177.6$ ms
  * Per-Frame Latency: $22.2$ ms
  * Processing Speed: **45.04 FPS** (**comfortably exceeds 30 FPS broadcast video requirements**)

---

## 4. How to Use the Model Directly in Python (Headless / Programmatic Inference)

To integrate the trained model into an automated backend, Python script, or API pipeline without the Gradio web UI:

```python
import torch
import cv2
import numpy as np
from src.models.temporal_vit import AdvancedTemporalViT
from src.preprocessing import FaceExtractor

# 1. Initialize model and load trained weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AdvancedTemporalViT(sequence_length=8, use_mocks=False).to(device)
model.classifier[-1] = torch.nn.Identity()
model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
model.eval()

extractor = FaceExtractor(device=device, image_size=224)

# 2. Extract 8 face frames from video
cap = cv2.VideoCapture("path/to/test_video.mp4")
face_frames = []
while cap.isOpened() and len(face_frames) < 8:
    ret, frame = cap.read()
    if not ret:
        break
    face = extractor.extract_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if face is not None:
        face_frames.append(face)
cap.release()

if len(face_frames) < 8:
    raise ValueError("Video must contain at least 8 detectable face frames.")

# 3. Construct spatial and optical residual difference tensors
x = torch.stack(face_frames).unsqueeze(0).to(device)  # Shape: (1, 8, 3, 224, 224)
residuals = [torch.zeros_like(x[:, 0])] + [torch.abs(x[:, t] - x[:, t - 1]) for t in range(1, 8)]
res = torch.stack(residuals, dim=1)  # Shape: (1, 8, 3, 224, 224)

# 4. Predict calibrated probability
with torch.no_grad():
    logits = model(x, res).squeeze(-1)
    prob = torch.sigmoid(logits).item()

verdict = "FAKE" if prob >= 0.5 else "REAL"
print(f"Prediction: {verdict} | Confidence: {prob * 100:.2f}%")
```

---

## 5. Model Checkpoint & Verification

The pre-trained model checkpoint (`best_model.pth`, 394.25 MB) is hosted on Google Drive:  
**Google Drive Weights Link:** [https://drive.google.com/file/d/1wKtAmNffix9zGmuD4RAfEVe7NoFFweNa/view?usp=drive_link](https://drive.google.com/file/d/1wKtAmNffix9zGmuD4RAfEVe7NoFFweNa/view?usp=drive_link)

To verify model parameter integrity and run a test forward pass:
```bash
python verify_submission.py
```
An interactive forensic dashboard is also available via:
```bash
python app.py
```
