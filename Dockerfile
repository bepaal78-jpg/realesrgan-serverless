# Schlankes Basis-Bild
VON Python: 3,10-schlank

# Arbeitsverzeichnis
ARBEITSVERZEICHNIS /App

# WICHTIG: build-essential hinzufügen für Python-Modul, die kompilierten werden müssen
# libgl1 für OpenCV
AUSFÜHREN apt-get update && apt-get install -y \
 bauessentiell \
 libgl1 \
 libglib2.0-0 \
 Git \
 wget \
 && rm -rf/var/lib/apt/lists/*

# 1. PyTorch CPU-Version (Wicht für die Große!)
AUSFÜHREN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Restliche Abhängigkeiten
# Wir nutzen --prefer-binary um Kompilier-Fehler zu vermischen
AUSFÜHREN pip install --no-cache-dir --prefer-binary \
 Runpod \
 Realesrgan \
 Grundlagen \
 facexlib \
 gfpgan \
 opencv-python-kopflos \
 Kissen \
 Numpy

# Modell herunterladen
AUSFÜHREN mkdir -p/app/models && \
 wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
 -O/app/models/RealESRGAN_x4plus.pth

# Handler-Skript kopieren
KOPIEREN handler.py/app/handler.py

# Start-Befehl: -u sorgt dafür, dass wir Logs sofort sehen (wichtig für Debugging!)
CMD ["Python", "-u", "handler.py"]
