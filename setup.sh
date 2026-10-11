#!/bin/bash
set -e

echo "Setting up Deepfake Detection Environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip

# Install PyTorch CPU only to save gigabytes of download!
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements (without torch to avoid overriding)
grep -v "^torch$" requirements.txt | grep -v "^torchvision$" > reqs_cpu.txt
pip install -r reqs_cpu.txt

echo "Environment setup complete."
