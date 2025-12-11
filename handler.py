importieren Runpod
importieren Basis64
importieren io
importieren Fackel
importieren Numpy als np
von PIL importieren Bild
importieren Lebenslauf
von Realesrgan importieren RealESRGANer
von basicsr.archs.rrdbnet_arch importieren RRDBNet

# --- Automatische Hardware-Erkennung ---
# Prüft, ob eine Nvidia GPU vergügbar ist. Wenn nicht, keine CPU.
wenn Fackel.Cuda.ist_verfügbar():
 Gerät_Typ = 'cuda'
 gpu_id = 0
 use_half = Wahr  # FP16 ist Schneller auf GPU
    drucken("🚀 Läuft auf GPU (CUDA)")
sonst:
 Gerät_Typ = 'cpu'
 gpu_id = Keine    # Keine zwingt RealESRGAN auf CPU
 use_half = Falsch # CPU unterstütz FP16 oft schlecht
    drucken(„⚠️ Läuft auf der CPU (langsamer, funktioniert aber ohne GPU)“)

# --- Modell beladen ---
drucken(„Real-ESRGAN-Modell wird geladen ...“)
Modell = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, Skala=4)

Upsampler = RealESRGANer(
 Skala=4,
 model_path='/app/models/RealESRGAN_x4plus.pth',
 Modell=Modell,
 Fliese=0,
 tile_pad=10,
 pre_pad=0,
 Hälfte=verwenden_halb, # Dynamisch je nach Hardware
 gpu_id=gpu_id # Dynamisch je nach Hardware
)
drucken(„Modell erfolgreich geladen!“)

def upscale_image(Job):
    """
 Handler für RunPod Serverless
 Eingabe: {"image": "base64_encoded_image", "scale": 4}
 Ausgabe: {"image": "base64_encoded_upscaled_image"}
 """
    versuchen:
 job_input = Job['Eingabe']
        
        # Base64 Bild dekodieren
 image_b64 = job_input.bekommen('Bild')
        # Standard-Scale auf 4 Setzen, fällt nicht gegengeben
 target_scale = job_input.bekommen('Skala', 4)
        
        wenn nicht Bild_b64:
            zurückgeben {"Fehler": „Kein Bild bereitgestellt“}
        
        # Base64 dekodieren
 image_data = base64.b64dekodieren(Bild_b64)
 Bild = Bild.öffnen(io.BytesIO(Bild_Daten))
        
        # Konvertiere zu numpy array (BGR für cv2)
 img_array = np.Array(Bild)
        
        # Farbkanäle korrigieren
        wenn Länge(img_array.Form) == 2:  # Graustufen
 img_array = cv2.cvtColor(img_array, cv2.FARBE_GRAY2BGR)
        Elif img_array.Form[2] == 4:  # RGBA
 img_array = cv2.cvtColor(img_array, cv2.FARBE_RGBA2BGR)
        Elif img_array.Form[2] == 3:  # RGB
 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Hochskalierung durchfürhren
        drucken(f"Bild hochskalieren... Größe: {img_array.Form}")
        
        # RealESRGAN für das Upscaling durch (outscale bestimmt den Zoom-Faktor)
 Ausgabe, _ = Upsampler.verbessern(img_array, outscale=target_scale)
        
        # Zum RGB für Kissen
 Ausgabe = cv2.cvtColor(Ausgabe, cv2.COLOR_BGR2RGB)
 output_image = Bild.Fromarray(Ausgabe)
        
        # Zu base64 kodieren
 gepuffert = io.BytesIO()
 Ausgabe_Bild.speichern(gepuffert, Format="PNG")
 img_str = base64.b64-codierung(gepuffert.Wert erhalten()).dekodieren()
        
        zurückgeben {
            "Bild": img_str,
            "original_size": Liste(Bild.Größe),
            "upscaled_size": Liste(Ausgabe_Bild.Größe),
            "Gerät_benutzt": Gerät_Typ
        }
        
    außer Ausnahme als e:
        drucken(f"Fehler: {str(e)}")
        zurückgeben {"Fehler": str(e)}

# RunPod Serverless Handler starten
Runpod.serverlos.starten({"Handler": upscale_image})
