import librosa
import matplotlib.pyplot as plt

canciones = [
    ("blues", "blues.00000.wav"),
    ("classical", "classical.00000.wav"),
    ("jazz", "jazz.00000.wav"),
    ("metal", "metal.00000.wav"),
]

for cancion in canciones:
    audio, sr = librosa.load(
        f"Data/genres_original/{cancion[0]}/{cancion[1]}",
        sr=None
    )

    print("Frecuencia de muestreo:", sr)
    print("Cantidad de muestras:", len(audio))
    print("Duración:", len(audio)/sr)

    plt.figure(figsize=(12,4))
    plt.plot(audio)
    plt.title(f"{cancion[0]}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.show()



