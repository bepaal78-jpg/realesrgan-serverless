FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Arbeitsverzeichnis
WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
RUN pip install --no-cache-dir \
    runpod \
    realesrgan \
    basicsr \
    facexlib \
    gfpgan \
    opencv-python-headless \
    Pillow

# Model herunterladen (beim Build)
RUN mkdir -p /app/models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O /app/models/RealESRGAN_x4plus.pth

# Handler Script
COPY handler.py /app/handler.py

# RunPod Handler starten
CMD ["python", "-u", "handler.py"]