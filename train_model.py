import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Random Forest es un algoritmo de aprendizaje supervisado basado en un conjunto
# de árboles de decisión. Durante el entrenamiento, cada árbol aprende patrones
# diferentes a partir de subconjuntos aleatorios de los datos y de las características.
#
# Para clasificar una nueva canción, cada árbol emite una predicción de género
# (blues, jazz, metal, etc.) y el bosque completo decide mediante votación mayoritaria.
#
# En este proyecto, el modelo recibe como entrada características extraídas mediante
# técnicas de procesamiento digital de señales (Spectral Centroid, Bandwidth, ZCR,
# RMS y MFCCs) y aprende a asociarlas con los distintos géneros musicales.
#
# Este enfoque permite evaluar si las características obtenidas del análisis
# tiempo-frecuencia contienen suficiente información para distinguir géneros musicales.

df = pd.read_csv("features.csv")

# Eliminar columnas que no son características

X = df.drop(
    columns=["genre", "file"]
)

y = df["genre"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Entrenar
modelo.fit(X_train, y_train)

# Predecir
y_pred = modelo.predict(X_test)

# Resultados
print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))