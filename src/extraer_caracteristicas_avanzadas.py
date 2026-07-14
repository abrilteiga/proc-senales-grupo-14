import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import pywt

# Ignorar advertencias de librosa
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "genres_original"
OUTPUT_PATH = BASE_DIR / "datasets_procesados" / "features_avanzadas.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extraer_de_segmento(audio_seg, sr, genero, archivo, seg_idx):
    try:
        # 1. Características básicas (Promedios) y de variabilidad (Desviaciones estándar)
        rms = librosa.feature.rms(y=audio_seg)
        zcr = librosa.feature.zero_crossing_rate(y=audio_seg)
        centroid = librosa.feature.spectral_centroid(y=audio_seg, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_seg, sr=sr)

        # 2. MFCCs (13 coeficientes promedio y std)
        mfccs = librosa.feature.mfcc(y=audio_seg, sr=sr, n_mfcc=13)

        # 3. Wavelets DWT (Descomposición en 5 niveles con db4 -> 6 subbandas)
        coeffs = pywt.wavedec(audio_seg, "db4", level=5)

        # Consolidar características del dominio del tiempo y de la STFT
        fila = {
            "genre": genero,
            "file": f"{archivo}_seg{seg_idx}",
            # Promedios de X(t)
            "centroid_mean": np.mean(centroid),
            "bandwidth_mean": np.mean(bandwidth),
            "zcr_mean": np.mean(zcr),
            "rms_mean": np.mean(rms),
            # Desviaciones estándar de X(t) (Variabilidad / Dinámica)
            "centroid_std": np.std(centroid),
            "bandwidth_std": np.std(bandwidth),
            "zcr_std": np.std(zcr),
            "rms_std": np.std(rms),
        }

        # Promedios y Stds de MFCCs (26)
        for i in range(13):
            fila[f"mfcc_mean_{i+1}"] = np.mean(mfccs[i])
            fila[f"mfcc_std_{i+1}"] = np.std(mfccs[i])

        # Características de Wavelet W(t) para cada una de las 6 subbandas:
        # Energía, Desviación Estándar, Cruces por Cero y Entropía de Shannon
        for i, coeff in enumerate(coeffs, 1):
            # Energía de W(t)
            fila[f"dwt_energy_{i}"] = np.mean(coeff**2)
            # Desviación estándar de W(t) (Variabilidad de subbanda)
            fila[f"dwt_std_{i}"] = np.std(coeff)
            # Cruces por cero de los coeficientes de subbanda
            fila[f"dwt_zcr_{i}"] = np.mean(np.diff(np.sign(coeff)) != 0)
            # Entropía de Shannon de los coeficientes (Desorden)
            abs_c = np.abs(coeff)
            sum_abs = np.sum(abs_c)
            if sum_abs > 0:
                p = abs_c / sum_abs
                entropy_val = -np.sum(p * np.log2(p + 1e-12))
            else:
                entropy_val = 0.0
            fila[f"dwt_entropy_{i}"] = entropy_val

        return fila
    except Exception as e:
        print(f"Error en segmento {seg_idx} de {archivo}: {e}")
        return None


def extraer_de_archivo(args):
    ruta_audio, genero, archivo = args
    filas = []
    try:
        audio, sr = librosa.load(ruta_audio, sr=None)
        # Dividimos en intervalos de 5 segundos
        chunk_size = 5 * sr
        num_chunks = int(len(audio) // chunk_size)

        for seg_idx in range(num_chunks):
            inicio = seg_idx * chunk_size
            fin = (seg_idx + 1) * chunk_size
            segmento = audio[inicio:fin]

            fila = extraer_de_segmento(segmento, sr, genero, archivo, seg_idx + 1)
            if fila is not None:
                filas.append(fila)
        return filas
    except Exception as e:
        print(f"Error procesando {genero}/{archivo}: {e}")
        return []


def obtener_audios():
    audios = []
    for carpeta_genero in sorted(DATASET_PATH.iterdir()):
        if not carpeta_genero.is_dir():
            continue

        genero = carpeta_genero.name
        for ruta_audio in sorted(carpeta_genero.glob("*.wav")):
            audios.append((ruta_audio, genero, ruta_audio.name))
    return audios


def main():
    audios = obtener_audios()
    print(f"Archivos encontrados: {len(audios)}")
    print("Iniciando extracción por segmentos de 5 segundos en paralelo...")

    resultados = []
    with ProcessPoolExecutor() as executor:
        for lista_filas in executor.map(extraer_de_archivo, audios):
            resultados.extend(lista_filas)

    # Guardar a CSV
    df = pd.DataFrame(resultados)
    df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"\nExtracción completada. Guardado en: {OUTPUT_PATH}"
    )
    print(f"Segmentos totales guardados: {len(df)}")
    print(f"Total características por segmento: {df.shape[1] - 2}")


if __name__ == "__main__":
    main()
