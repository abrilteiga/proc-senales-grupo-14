import librosa
import numpy as np
import pandas as pd

canciones = [
    ("blues", "blues.00000.wav"),
    ("classical", "classical.00000.wav"),
    ("jazz", "jazz.00000.wav"),
    ("metal", "metal.00000.wav"),
]

resultados = []

for genero, archivo in canciones:

    audio, sr = librosa.load(
        f"Data/genres_original/{genero}/{archivo}",
        sr=None
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    ) #centro de masa del espectro / + alto = + ruido

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    ) # ancho de banda + alto = sonido complejo / + alto = sonido + concentrado

    #cuantas veces la señal cruzo el cero
    # + alto = más ruidos
    # + bajo = + armonico
    zcr = librosa.feature.zero_crossing_rate(audio) 

    #Mide la potencia promedio
    rms = librosa.feature.rms(y=audio)

    resultados.append({
        "Genero": genero,
        "Spectral Centroid": np.mean(spectral_centroid),
        "Bandwidth": np.mean(spectral_bandwidth),
        "Zero Crossing Rate": np.mean(zcr),
        "RMS Energy": np.mean(rms)
    })

df = pd.DataFrame(resultados)

print(df)