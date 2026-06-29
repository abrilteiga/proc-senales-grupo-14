import os

import matplotlib.pyplot as plt
import pandas as pd


# ==========================
# Configuración
# ==========================

FEATURES_FILE = "features.csv"
OUTPUT_DIR = "graficos_caracteristicas"

# Crear la carpeta donde se guardarán los gráficos
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# Cargar datos
# ==========================

df = pd.read_csv(FEATURES_FILE)

print("Cantidad de canciones:", len(df))
print("Columnas disponibles:")
print(df.columns.tolist())

print("\nCantidad de canciones por género:")
print(df["genre"].value_counts())


# ==========================
# Estadísticas por género
# ==========================

columnas_basicas = [
    "centroid",
    "bandwidth",
    "zcr",
    "rms"
]

promedios = df.groupby("genre")[columnas_basicas].mean()
desviaciones = df.groupby("genre")[columnas_basicas].std()

print("\nPromedios por género:")
print(promedios)

print("\nDesviaciones estándar por género:")
print(desviaciones)

promedios.to_csv(
    os.path.join(OUTPUT_DIR, "promedios_por_genero.csv")
)

desviaciones.to_csv(
    os.path.join(OUTPUT_DIR, "desviaciones_por_genero.csv")
)


# ==========================
# Gráficos de promedios
# ==========================

for caracteristica in columnas_basicas:

    promedio_caracteristica = (
        df.groupby("genre")[caracteristica]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(11, 6))

    promedio_caracteristica.plot(
        kind="bar"
    )

    plt.title(
        f"Promedio de {caracteristica} por género"
    )
    plt.xlabel("Género")
    plt.ylabel(caracteristica)
    plt.xticks(rotation=45)
    plt.tight_layout()

    ruta_salida = os.path.join(
        OUTPUT_DIR,
        f"promedio_{caracteristica}.png"
    )

    plt.savefig(ruta_salida)
    plt.show()
    plt.close()


# ==========================
# Diagramas de caja
# ==========================

# Los boxplots permiten observar no solo el promedio,
# sino también la dispersión y los valores atípicos.

for caracteristica in columnas_basicas:

    generos = sorted(df["genre"].unique())

    datos_por_genero = [
        df[df["genre"] == genero][caracteristica]
        for genero in generos
    ]

    plt.figure(figsize=(12, 6))

    plt.boxplot(
        datos_por_genero,
        tick_labels=generos,
        showfliers=True
    )

    plt.title(
        f"Distribución de {caracteristica} por género"
    )
    plt.xlabel("Género")
    plt.ylabel(caracteristica)
    plt.xticks(rotation=45)
    plt.tight_layout()

    ruta_salida = os.path.join(
        OUTPUT_DIR,
        f"distribucion_{caracteristica}.png"
    )

    plt.savefig(ruta_salida)
    plt.show()
    plt.close()


# ==========================
# Análisis de los MFCC
# ==========================

columnas_mfcc = [
    columna
    for columna in df.columns
    if columna.startswith("mfcc_")
]

if columnas_mfcc:

    promedios_mfcc = (
        df.groupby("genre")[columnas_mfcc]
        .mean()
    )

    promedios_mfcc.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "promedios_mfcc_por_genero.csv"
        )
    )

    # Graficar cada MFCC por separado
    for mfcc in columnas_mfcc:

        promedio_mfcc = (
            df.groupby("genre")[mfcc]
            .mean()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(11, 6))

        promedio_mfcc.plot(
            kind="bar"
        )

        plt.title(
            f"Promedio de {mfcc} por género"
        )
        plt.xlabel("Género")
        plt.ylabel(mfcc)
        plt.xticks(rotation=45)
        plt.tight_layout()

        ruta_salida = os.path.join(
            OUTPUT_DIR,
            f"promedio_{mfcc}.png"
        )

        plt.savefig(ruta_salida)
        plt.close()

    print(
        f"\nSe analizaron {len(columnas_mfcc)} coeficientes MFCC."
    )

else:
    print("\nNo se encontraron columnas MFCC.")


print(
    f"\nLos gráficos fueron guardados en: {OUTPUT_DIR}"
)