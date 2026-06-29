import librosa
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt

# ==========================
# Cargar audio
# ==========================

audio, sr = librosa.load(
    "Data/genres_original/metal/metal.00000.wav",
    sr=None
)

# ==========================
# Crear filtro pasa bajos
# ==========================

frecuencia_corte = 2000  # Hz

nyquist = sr / 2

b, a = butter(
    N=4,
    Wn=frecuencia_corte / nyquist,
    btype="low"
)

audio_filtrado = filtfilt(
    b,
    a,
    audio
)

# ==========================
# Graficar
# ==========================

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(audio)
plt.title("Audio original")

plt.subplot(2,1,2)
plt.plot(audio_filtrado)
plt.title("Audio filtrado (Pasa bajos)")

plt.tight_layout()
plt.show()