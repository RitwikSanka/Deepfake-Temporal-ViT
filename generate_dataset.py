import os
import cv2
import numpy as np

def generate_mock_videos(output_dir, num_real=10, num_fake=10):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Real Videos
    for i in range(num_real):
        path = os.path.join(output_dir, f"real_{i}.mp4")
        create_video(path, is_fake=False)
        print(f"Generated {path}")
        
    # Generate Fake Videos
    for i in range(num_fake):
        path = os.path.join(output_dir, f"fake_{i}.mp4")
        create_video(path, is_fake=True)
        print(f"Generated {path}")

def create_video(path, is_fake):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 15.0, (640, 480))
    
    # Create a base face representation
    center_x, center_y = 320, 240
    
    for frame_idx in range(15): # 1 second video
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Micro-expression/motion simulation
        # Real videos have smooth motion, fake videos have artificial jitters
        if is_fake:
            jitter = np.random.randint(-10, 10)
        else:
            jitter = int(5 * np.sin(frame_idx * 0.5))
            
        # Draw face
        cv2.circle(frame, (center_x + jitter, center_y + jitter), 100, (220, 180, 150), -1)
        # Eyes
        cv2.circle(frame, (center_x - 30 + jitter, center_y - 20 + jitter), 15, (255, 255, 255), -1)
        cv2.circle(frame, (center_x + 30 + jitter, center_y - 20 + jitter), 15, (255, 255, 255), -1)
        # Pupils
        cv2.circle(frame, (center_x - 30 + jitter, center_y - 20 + jitter), 5, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + 30 + jitter, center_y - 20 + jitter), 5, (0, 0, 0), -1)
        
        if is_fake:
            # Add synthetic blending artifacts (random noise boxes on the "face")
            noise_x = center_x + np.random.randint(-50, 50)
            noise_y = center_y + np.random.randint(-50, 50)
            frame[noise_y:noise_y+20, noise_x:noise_x+20] = np.random.randint(0, 255, (20, 20, 3))
            
        out.write(frame)
        
    out.release()

if __name__ == "__main__":
    generate_mock_videos("./data", num_real=5, num_fake=5)
