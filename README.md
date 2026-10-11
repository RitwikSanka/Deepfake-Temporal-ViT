# Dual-Stream Temporal Vision Transformer (ViT) for Deepfake Detection

**Institution:** Department of Electrical Engineering, Indian Institute of Technology Madras (IIT Madras)  
**Contact Email:** [ee23b149@smail.iitm.ac.in](mailto:ee23b149@smail.iitm.ac.in)

An end-to-end, production-grade deep learning system for real-time deepfake video detection. This project captures subtle temporal micro-expression inconsistencies and face-swapping blending artifacts across video sequences by fusing a pre-trained spatial Vision Transformer with a lightweight optical residual difference stream, multi-head temporal self-attention, and a bidirectional GRU.

---

## 1. System Architecture & Data Flow

```
Input Video (8 Frames, 224x224)
  ├──> Spatial Stream: Pre-trained ViT-B/16 (CLS Tokens: 8 x 768)
  └──> Residual Stream: Inter-Frame Differences |Xt - Xt-1| ──> Shallow CNN (8 x 768)
                               │
                               ▼
                   Cross-Stream Linear Fusion (Projects 1536 -> 768)
                               │
                               ▼
                 Sinusoidal Positional Encodings
                               │
                               ▼
                 Multi-Head Temporal Self-Attention (2 Layers)
                               │
                               ▼
                 Bidirectional GRU (Temporal Aggregation)
                               │
                               ▼
                 MLP Classifier ──> Sigmoid Deepfake Probability [0.0 - 1.0]
```

---

## 2. Key Empirical Results (Celeb-DF v2 Benchmark)

The model was fine-tuned and validated on the challenging **Celeb-DF v2** dataset using an Nvidia T4 GPU:

* **ROC AUC:** `0.8834` (88.34%)
* **Validation Accuracy:** `82.00%`
* **Deepfake Recall (Sensitivity):** `86.54%` (90/104 deepfakes correctly detected)
* **Authentic Video Specificity:** `77.08%` (74/96 real videos correctly verified)
* **F1-Score:** `0.8334`
* **Equal Error Rate (EER):** `0.1620`
* **Inference Throughput (GPU):** `45.04 FPS` (exceeds 30 FPS broadcast streaming requirements)

---

## 3. How to Use the Model Directly in Python (Headless / Programmatic API)

To run inference directly in your own Python scripts, backend services, or batch evaluation loops without launching the Gradio web application:

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

## 4. Directory Structure

```
deepfake_detection/
├── src/
│   ├── models/
│   │   └── temporal_vit.py     # Dual-Stream Temporal ViT implementation
│   ├── preprocessing.py        # MTCNN / Haar face extraction & ImageNet normalization
│   ├── dataset.py              # PyTorch Dataset for video and image frame sequences
│   └── visualize.py            # ViT Attention Rollout & Grad-CAM visualizer
├── weights/
│   └── best_model.pth          # Fine-tuned checkpoint on Celeb-DF v2 (394 MB)
├── figures/
│   ├── roc_curve.png           # Actual ROC curve plot (AUC = 0.8834)
│   ├── confusion_matrix.png    # Actual confusion matrix (82.00% Accuracy)
│   ├── attention_maps.png      # Attention Rollout spatial attribution heatmaps
│   └── benchmark_table.csv     # Quantitative comparison with ablated baselines
├── report/
│   ├── report.pdf              # Publication-grade academic PDF report (WeasyPrint)
│   ├── FINAL_REPORT.md         # Executive markdown summary & profiling tables
│   └── confidence_distribution.png
├── Colab_Training_Master.ipynb # Master Colab training notebook with Drive integration
├── app.py                      # Interactive Gradio web application
├── benchmark.py                # Local comparative evaluation suite
├── verify_submission.py        # 1-click integrity and audit verification script
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation and handover blueprint
```

---

## 5. Quick Start & Verification

### Step 1: Run the Submission Integrity Audit
To verify that all weights, figures, and model components are intact and functional:
```bash
python verify_submission.py
```

### Step 2: Launch the Interactive Live Web Demo
To launch the interactive Gradio web application for live video analysis:
```bash
python app.py
```
Open `http://localhost:7860` in your web browser, upload any video, and see the real-time classification verdict and temporal probability curve!

### Step 3: Reproduce Training
To train the model from scratch or inspect the Colab training pipeline:
Open `Colab_Training_Master.ipynb` in Google Colab, select a T4 GPU runtime, and run all cells.

---

## 6. Model Checkpoint (Google Drive)

The fine-tuned model checkpoint (`best_model.pth`, 394.25 MB) is hosted on Google Drive:  
**Google Drive Weights Link:** [https://drive.google.com/file/d/1wKtAmNffix9zGmuD4RAfEVe7NoFFweNa/view?usp=drive_link](https://drive.google.com/file/d/1wKtAmNffix9zGmuD4RAfEVe7NoFFweNa/view?usp=drive_link)
