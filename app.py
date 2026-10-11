import gradio as gr
import torch
import cv2
import numpy as np

from src.models.temporal_vit import AdvancedTemporalViT
from src.preprocessing import FaceExtractor

import os

# Initialize globally
device = "cuda" if torch.cuda.is_available() else "cpu"
seq_len = 8
model = AdvancedTemporalViT(sequence_length=seq_len, use_mocks=False).to(device)

weights_path = "weights/best_model.pth"
has_custom_weights = False
if os.path.exists(weights_path):
    print(f"Loading trained weights from {weights_path}...")
    try:
        model.classifier[-1] = torch.nn.Identity()
        model.load_state_dict(torch.load(weights_path, map_location=device))
        has_custom_weights = True
        print("Trained weights successfully loaded into Temporal ViT!")
    except Exception as e:
        print(f"Error loading {weights_path}: {e}")
        model.classifier[-1] = torch.nn.Sigmoid()
        has_custom_weights = False
else:
    print(f"Notice: {weights_path} not found. Running with foundation ViT weights.")
    print("Download best_model.pth from Google Drive and place in weights/ for custom trained predictions.")

model.eval()
extractor = FaceExtractor(device=device, image_size=224)

def analyze_video(video_path, progress=gr.Progress(track_tqdm=False)):
    if not video_path:
        return "Please upload a video file.", None
        
    # Handle both gr.File (which may pass a file object/dict) and gr.Video (filepath string)
    if isinstance(video_path, dict) and 'name' in video_path:
        video_path = video_path['name']
    elif hasattr(video_path, 'name'):
        video_path = video_path.name
        
    progress(0.1, desc="Opening video and extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Extract up to 16 frames for fast, responsive live demo
    while cap.isOpened() and len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_tensor = extractor.extract_face(frame_rgb)
        
        if face_tensor is not None:
            frames.append(face_tensor)
            progress(0.1 + 0.4 * (len(frames) / 16.0), desc=f"Extracted face frame {len(frames)}/16...")
            
    cap.release()
    
    if len(frames) < seq_len:
        return f"Could not extract enough face frames (minimum {seq_len} required, found {len(frames)}).", None
        
    progress(0.6, desc="Constructing temporal sliding windows...")
    sequences = []
    for i in range(len(frames) - seq_len + 1):
        seq = torch.stack(frames[i:i+seq_len])
        sequences.append(seq)
        
    sequences_tensor = torch.stack(sequences).to(device)
    
    progress(0.75, desc="Computing inter-frame optical residuals...")
    residuals = [torch.zeros_like(sequences_tensor[:, 0])] + [torch.abs(sequences_tensor[:, t] - sequences_tensor[:, t-1]) for t in range(1, seq_len)]
    residuals_tensor = torch.stack(residuals, dim=1)
    
    progress(0.85, desc="Evaluating Temporal ViT model...")
    with torch.no_grad():
        out = model(sequences_tensor, residuals_tensor).squeeze(-1)
        if has_custom_weights:
            probs = torch.sigmoid(out).cpu().numpy()
        else:
            probs = out.cpu().numpy()
        
    progress(1.0, desc="Done!")
    probs = np.atleast_1d(probs)
    avg_prob = float(np.mean(probs))
    verdict = "FAKE" if avg_prob >= 0.5 else "REAL"
    color = "red" if verdict == "FAKE" else "green"
    
    html_result = f"<h1 style='color: {color}; text-align: center;'>{verdict}</h1>"
    html_result += f"<h3 style='text-align: center;'>Confidence: {avg_prob*100:.2f}%</h3>"
    
    # Create temporal plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 3))
    plt.plot(range(len(probs)), probs, marker='o', linestyle='-', color='purple')
    plt.axhline(0.5, color='black', linestyle='--')
    plt.ylim(0, 1)
    plt.title("Deepfake Probability Over Time")
    plt.xlabel("Sequence Window")
    plt.ylabel("Probability (1 = Fake)")
    plt.tight_layout()
    
    return html_result, fig

with gr.Blocks(title="Real-Time Deepfake Detector") as app:
    gr.Markdown("# Temporal Vision Transformer (ViT) Deepfake Detector")
    gr.Markdown("Upload any video or dataset clip to analyze micro-expression inconsistencies and blending artifacts.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Upload Video File (All Codecs: Celeb-DF / MP4 / AVI)"):
                    file_in = gr.File(label="Upload Any Video File (Bypasses Browser Codec Issues)", file_types=["video", ".mp4", ".avi", ".mov", ".mkv"])
                    analyze_file_btn = gr.Button("Analyze Video File", variant="primary")
                with gr.TabItem("Browser Video Player (H.264 Only)"):
                    video_in = gr.Video(label="Browser Playable Video")
                    analyze_vid_btn = gr.Button("Analyze Video Player", variant="secondary")
            
        with gr.Column(scale=1):
            result_html = gr.HTML(label="Verdict")
            temporal_plot = gr.Plot(label="Analysis Over Time")
            
    analyze_file_btn.click(
        fn=analyze_video,
        inputs=[file_in],
        outputs=[result_html, temporal_plot]
    )
    analyze_vid_btn.click(
        fn=analyze_video,
        inputs=[video_in],
        outputs=[result_html, temporal_plot]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
