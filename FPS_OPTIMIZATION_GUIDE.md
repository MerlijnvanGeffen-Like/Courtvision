# FPS Optimalisatie Guide - Courtvision

## PC Instellingen om FPS te Verhogen

### 1. GPU/CUDA Gebruik (Belangrijkste!)

Als je een NVIDIA GPU hebt, zorg ervoor dat CUDA correct is geïnstalleerd:

**Check of CUDA werkt:**
```python
# Run dit in Python om te checken
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**CUDA Installeren (als je NVIDIA GPU hebt):**
1. Download CUDA Toolkit van: https://developer.nvidia.com/cuda-downloads
2. Download cuDNN van: https://developer.nvidia.com/cudnn
3. Installeer PyTorch met CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

**GPU Driver updaten:**
- Ga naar NVIDIA website → Drivers
- Download laatste Game Ready Driver voor je GPU model
- Installeer en herstart PC

---

### 2. Windows Power Settings

**High Performance Mode activeren:**
1. Ga naar: Control Panel → Power Options
2. Selecteer "High Performance" plan
3. Of klik op "Change plan settings" → "Change advanced power settings"
4. Zet "Processor power management" → "Minimum processor state" op 100%
5. Zet "Maximum processor state" op 100%

**Via Command Prompt (Admin):**
```cmd
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

---

### 3. Windows Graphics Settings

**Hardware Acceleration:**
1. Ga naar: Settings → System → Display → Graphics settings
2. Zet "Hardware-accelerated GPU scheduling" AAN
3. Herstart PC

**GPU Priority voor Python:**
1. Ga naar: Settings → System → Display → Graphics settings
2. Klik "Browse" en voeg `python.exe` toe (meestal in Python installatie folder)
3. Zet op "High performance" (gebruik GPU)
4. Herhaal voor `pythonw.exe`

---

### 4. NVIDIA Control Panel (voor NVIDIA GPU gebruikers)

**Global Settings:**
1. Rechtsklik op desktop → NVIDIA Control Panel
2. 3D Settings → Manage 3D Settings → Global Settings:
   - **Power management mode**: Prefer maximum performance
   - **Texture filtering - Quality**: Performance
   - **Threaded optimization**: On
   - **Preferred graphics processor**: High-performance NVIDIA processor

**Program Settings (voor Python):**
1. 3D Settings → Manage 3D Settings → Program Settings
2. Add → Select `python.exe`
3. Zet "Preferred graphics processor" op: High-performance NVIDIA processor
4. Power management mode: Prefer maximum performance

---

### 7. Python/Anaconda Optimalisatie

**Python versie:**
- Gebruik Python 3.9 of 3.10 (nieuwere versies kunnen trager zijn)
- 64-bit versie gebruiken (niet 32-bit)

**Anaconda/Conda optimalisatie:**
```bash
# Zorg dat je de juiste PyTorch versie hebt
# Of pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

**Als FPS nog steeds laag is:**

1. **Check welke device wordt gebruikt:**
   - Kijk in console output bij start: "Running on CUDA" of "Using CPU"
   - Als het CPU is maar je hebt GPU: CUDA niet goed geïnstalleerd

2. **Test met kleinere image size:**
   - In code: `self.imgsz = 256` of zelfs `224` (maar accuratesse kan afnemen)

3. **Camera resolutie verlagen:**
   - In code: `self.capture_width = 320`, `self.capture_height = 240`

4. **Check andere applicaties:**
   - Sluit browsers, games, video players
   - Sluit onnodige background apps

---

## Verwacht FPS

**Met GPU (CUDA):**
- Goede GPU (RTX 3060+): 25-35 FPS
- Mid-range GPU (GTX 1660): 20-30 FPS
- Oudere GPU: 15-25 FPS

**Zonder GPU (CPU only):**
- Moderne CPU (i7/i9): 5-15 FPS
- Oudere CPU: 3-10 FPS

**Tip:** GPU geeft meestal 3-5x snellere performance dan CPU!
