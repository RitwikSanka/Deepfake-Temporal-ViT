import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from src.preprocessing import FaceExtractor

class DeepfakeVideoDataset(Dataset):
    """
    Loads face video sequences, performs MTCNN cropping/normalization, 
    and returns both RGB frames and temporal residual frames.
    """
    def __init__(self, video_paths, labels, sequence_length=8, image_size=224, augment=False, device='cpu'):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment
        self.extractor = FaceExtractor(device=device, image_size=image_size)

    def __len__(self):
        return len(self.video_paths)
        
    def _apply_motion_jitter(self, frames):
        """
        Synthetic motion-jitter/artifact augmentation module to stress-test 
        micro-expression temporal consistency.
        """
        aug_frames = []
        for frame in frames:
            if random.random() < 0.3:
                # Add random artifact block (simulate blending artifacts)
                h, w, _ = frame.shape
                x = random.randint(0, w-20)
                y = random.randint(0, h-20)
                frame = frame.copy()
                frame[y:y+20, x:x+20] = np.random.randint(0, 255, (20, 20, 3))
            
            if random.random() < 0.3:
                # Motion jitter by minor shifting
                shift_x = random.randint(-5, 5)
                shift_y = random.randint(-5, 5)
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            aug_frames.append(frame)
        return aug_frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = []
        import os
        import glob
        if os.path.isdir(video_path):
            img_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")) + glob.glob(os.path.join(video_path, "*.png")))
            for p in img_paths[:self.sequence_length * 2]:
                frame = cv2.imread(p)
                if frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened() and len(frames) < self.sequence_length * 2: 
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        
        if self.augment:
            frames = self._apply_motion_jitter(frames)
            
        face_tensors = []
        for frame in frames:
            face_t = self.extractor.extract_face(frame)
            if face_t is not None:
                face_tensors.append(face_t)
            if len(face_tensors) == self.sequence_length:
                break
                
        # Padding if sequence is too short
        while len(face_tensors) < self.sequence_length:
            if len(face_tensors) > 0:
                face_tensors.append(face_tensors[-1])
            else:
                face_tensors.append(torch.zeros((3, self.image_size, self.image_size)))
                
        x = torch.stack(face_tensors) # (seq_len, 3, 224, 224)
        
        # Compute temporal residual frames (optical difference)
        residuals = []
        for i in range(self.sequence_length):
            if i == 0:
                residuals.append(torch.zeros_like(x[0]))
            else:
                residuals.append(torch.abs(x[i] - x[i-1]))
        residuals = torch.stack(residuals)
        
        return x, residuals, torch.tensor(label, dtype=torch.float32)
