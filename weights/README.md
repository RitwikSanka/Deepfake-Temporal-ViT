# Pre-trained Model Weights

Due to GitHub's 100 MB file size limit, the trained model checkpoint (`best_model.pth`, 394.25 MB) is hosted on Google Drive:

- **Google Drive Download Link:** [best_model.pth Checkpoint](https://drive.google.com/file/d/1wKtAmNffix9zGmuD4RAfEVe7NoFFweNa/view?usp=drive_link)

### Setup Instructions
1. Download `best_model.pth` from the link above.
2. Place `best_model.pth` directly inside this `weights/` directory.
3. Run verification to confirm parameter integrity and inference:
   ```bash
   python verify_submission.py
   ```
