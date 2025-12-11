# Wir nutzen ein sehr schlankes Basis-Image
FROM python:3.10-slim

# Arbeitsverzeichnis
WORKDIR /app

# System Dependencies installieren
# WICHTIG: "libgl1" statt "libgl1-mesa-glx" für neuere Debian-Versionen
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 1. PyTorch CPU-Version installieren (Spart Speicherplatz!)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Restliche Dependencies installieren
RUN pip install --no-cache-dir \
    runpod \
    realesrgan \
    basicsr \
    facexlib \
    gfpgan \
    opencv-python-headless \
    Pillow \
    numpy

# Model herunterladen
RUN mkdir -p /app/models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O /app/models/RealESRGAN_x4plus.pth

# Handler Script kopieren
COPY handler.py /app/handler.py

# RunPod Handler starten
CMD ["python", "-u", "handler.py"]