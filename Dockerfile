# Schlankes Basis-Image
FROM python:3.10-slim

# Arbeitsverzeichnis
WORKDIR /app

# WICHTIG: build-essential hinzufügen für Python-Module, die kompiliert werden müssen
# libgl1 für OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 1. PyTorch CPU-Version (Wichtig für die Größe!)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Restliche Dependencies
# Wir nutzen --prefer-binary um Kompilier-Fehler zu vermeiden
RUN pip install --no-cache-dir --prefer-binary \
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

# Start-Befehl: -u sorgt dafür, dass wir Logs sofort sehen (wichtig für Debugging!)
CMD ["python", "-u", "handler.py"]