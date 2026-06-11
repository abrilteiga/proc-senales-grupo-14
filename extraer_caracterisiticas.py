import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "Data/genres_original"

resultados = []

for genero in os.listdir(DATASET_PATH):

    carpeta_genero = os.path.join(
        DATASET_PATH,
        genero
    )

    if not os.path.isdir(carpeta_genero):
        continue

    print(f"Procesando {genero}...")

    for archivo in os.listdir(carpeta_genero):

        if not archivo.endswith(".wav"):
            continue

        ruta = os.path.join(
            carpeta_genero,
            archivo
        )

        try:

            audio, sr = librosa.load(
                ruta,
                sr=None
            )

            # Características básicas

            centroid = np.mean(
                librosa.feature.spectral_centroid(
                    y=audio,
                    sr=sr
                )
            )

            bandwidth = np.mean(
                librosa.feature.spectral_bandwidth(
                    y=audio,
                    sr=sr
                )
            )

            zcr = np.mean(
                librosa.feature.zero_crossing_rate(
                    audio
                )
            )

            rms = np.mean(
                librosa.feature.rms(
                    y=audio
                )
            )

            # MFCC

            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13
            )

            mfcc_means = np.mean(
                mfccs,
                axis=1
            )

            fila = {
                "genre": genero,
                "file": archivo,
                "centroid": centroid,
                "bandwidth": bandwidth,
                "zcr": zcr,
                "rms": rms,
            }

            for i in range(13):
                fila[f"mfcc_{i+1}"] = mfcc_means[i]

            resultados.append(fila)

        except Exception as e:
            print(
                f"Error en {archivo}: {e}"
            )

df = pd.DataFrame(resultados)

print(df.head())

df.to_csv(
    "features.csv",
    index=False
)

print("\nCSV generado correctamente.")
print("Cantidad de canciones:", len(df))