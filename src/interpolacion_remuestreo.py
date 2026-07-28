"""Experimento de interpolacion y remuestreo aplicado a una senal musical.

Relacion con Clase_Interpolacion_PSUBA.pdf: se parte de una senal con
muestreo regular, se disminuye la frecuencia de muestreo y se estiman las
muestras intermedias con interpolacion lineal y cubica.
"""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample_poly


BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_AUDIO = BASE_DIR / "Data" / "genres_original" / "jazz" / "jazz.00000.wav"
OUTPUT_DIR = BASE_DIR / "outputs" / "interpolacion_remuestreo"
FACTOR_REDUCCION = 4
SEGUNDOS_GRAFICO = 0.025
SEGUNDOS_METRICAS = 5


def espectro_magnitud(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve el espectro de magnitud normalizado de una ventana de audio."""
    ventana = audio * np.hanning(len(audio))
    magnitud = np.abs(np.fft.rfft(ventana))
    return np.fft.rfftfreq(len(ventana), d=1 / sr), magnitud / (np.max(magnitud) + 1e-12)


def metricas(referencia: np.ndarray, estimacion: np.ndarray) -> dict[str, float]:
    error = referencia - estimacion
    rmse = float(np.sqrt(np.mean(error**2)))
    snr_db = float(10 * np.log10(np.mean(referencia**2) / (np.mean(error**2) + 1e-12)))
    return {"RMSE": rmse, "SNR_dB": snr_db}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(ARCHIVO_AUDIO, sr=None, mono=True, duration=SEGUNDOS_METRICAS)
    audio = audio - np.mean(audio)

    # resample_poly incluye un pasa-bajos antialias antes de bajar la tasa.
    audio_baja = resample_poly(audio, up=1, down=FACTOR_REDUCCION)
    sr_baja = sr / FACTOR_REDUCCION
    tiempos_bajos = np.arange(len(audio_baja)) / sr_baja
    tiempos_originales = np.arange(len(audio)) / sr

    # Se reconstruyen las muestras en la grilla temporal original.
    lineal = interp1d(tiempos_bajos, audio_baja, kind="linear", bounds_error=False, fill_value="extrapolate")
    cubica = interp1d(tiempos_bajos, audio_baja, kind="cubic", bounds_error=False, fill_value="extrapolate")
    audio_lineal = lineal(tiempos_originales)
    audio_cubica = cubica(tiempos_originales)

    resultados = pd.DataFrame([
        {"metodo": "interpolacion_lineal", **metricas(audio, audio_lineal)},
        {"metodo": "interpolacion_cubica", **metricas(audio, audio_cubica)},
    ])
    resultados.to_csv(OUTPUT_DIR / "metricas_remuestreo.csv", index=False)

    limite = int(SEGUNDOS_GRAFICO * sr)
    limite_bajo = int(SEGUNDOS_GRAFICO * sr_baja)
    plt.figure(figsize=(12, 7))
    plt.plot(tiempos_originales[:limite], audio[:limite], color="black", linewidth=1.3, label=f"Original ({sr / 1000:.2f} kHz)")
    plt.plot(tiempos_originales[:limite], audio_lineal[:limite], color="tab:orange", alpha=0.85, label="Reconstruccion lineal")
    plt.plot(tiempos_originales[:limite], audio_cubica[:limite], color="tab:blue", alpha=0.85, label="Reconstruccion cubica")
    plt.scatter(tiempos_bajos[:limite_bajo], audio_baja[:limite_bajo], color="tab:red", s=12, zorder=3, label=f"Muestras a {sr_baja} Hz")
    plt.title("Interpolacion despues de reducir la frecuencia de muestreo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interpolacion_temporal.png", dpi=160)
    plt.close()

    freqs_original, espec_original = espectro_magnitud(audio, sr)
    freqs_lineal, espec_lineal = espectro_magnitud(audio_lineal, sr)
    freqs_cubica, espec_cubica = espectro_magnitud(audio_cubica, sr)
    plt.figure(figsize=(12, 5))
    limite_hz = sr_baja / 2
    plt.plot(freqs_original, espec_original, color="black", label="Original")
    plt.plot(freqs_lineal, espec_lineal, color="tab:orange", alpha=0.8, label="Lineal")
    plt.plot(freqs_cubica, espec_cubica, color="tab:blue", alpha=0.8, label="Cubica")
    plt.axvline(limite_hz, color="tab:red", linestyle="--", label=f"Nyquist de la senal reducida ({limite_hz:.0f} Hz)")
    plt.title("Efecto del remuestreo en el dominio de la frecuencia")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud normalizada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interpolacion_espectro.png", dpi=160)
    plt.close()

    print(f"Audio original: {sr} Hz | audio reducido: {sr_baja} Hz")
    print(resultados.to_string(index=False))
    print(f"Salidas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
