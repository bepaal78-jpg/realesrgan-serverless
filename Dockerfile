FROM python:3.10-slim

WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies (CPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefer-binary \
    runpod \
    realesrgan \
    basicsr \
    facexlib \
    gfpgan \
    opencv-python-headless \
    Pillow \
    numpy

# Model download
RUN mkdir -p /app/models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O /app/models/RealESRGAN_x4plus.pth

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
