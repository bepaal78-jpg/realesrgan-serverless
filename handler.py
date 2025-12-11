import runpod
import base64
import io
import torch
import numpy as np
from PIL import Image
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- Automatische Hardware-Erkennung ---
# Prüft, ob eine Nvidia GPU verfügbar ist. Wenn nicht, nutze CPU.
if torch.cuda.is_available():
    device_type = 'cuda'
    gpu_id = 0
    use_half = True  # FP16 ist schneller auf GPU
    print("🚀 Running on GPU (CUDA)")
else:
    device_type = 'cpu'
    gpu_id = None    # None zwingt RealESRGAN auf CPU
    use_half = False # CPU unterstützt FP16 oft schlecht
    print("⚠️ Running on CPU (Slower but works without GPU)")

# --- Model laden ---
print("Loading Real-ESRGAN model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path='/app/models/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=use_half,  # Dynamisch je nach Hardware
    gpu_id=gpu_id   # Dynamisch je nach Hardware
)
print("Model loaded successfully!")

def upscale_image(job):
    """
    Handler für RunPod Serverless
    Input: {"image": "base64_encoded_image", "scale": 4}
    Output: {"image": "base64_encoded_upscaled_image"}
    """
    try:
        job_input = job['input']
        
        # Base64 Image dekodieren
        image_b64 = job_input.get('image')
        # Standard-Scale auf 4 setzen, falls nicht angegeben
        target_scale = job_input.get('scale', 4)
        
        if not image_b64:
            return {"error": "No image provided"}
        
        # Decode base64
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Konvertiere zu numpy array (BGR für cv2)
        img_array = np.array(image)
        
        # Farbkanäle korrigieren
        if len(img_array.shape) == 2:  # Graustufen
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Upscaling durchführen
        print(f"Upscaling image... Size: {img_array.shape}")
        
        # RealESRGAN führt das Upscaling durch (outscale bestimmt den Zoom-Faktor)
        output, _ = upsampler.enhance(img_array, outscale=target_scale)
        
        # Zurück zu RGB für Pillow
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output)
        
        # Encode zu base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "original_size": list(image.size),
            "upscaled_size": list(output_image.size),
            "device_used": device_type
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# RunPod Serverless Handler starten
runpod.serverless.start({"handler": upscale_image})