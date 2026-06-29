import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#model SVM para comparar resultados con el randomForest

# Support Vector Machine (SVM) es un algoritmo de aprendizaje supervisado
# que busca encontrar el hiperplano que mejor separa las clases en el espacio
# de características.
#
# En este trabajo, cada canción se representa mediante un conjunto de
# características extraídas del procesamiento digital de señales
# (MFCC, Spectral Centroid, Bandwidth, ZCR, RMS, etc.).
#
# El kernel RBF permite modelar fronteras de decisión no lineales,
# adecuadas para problemas donde los géneros musicales no son separables
# mediante una línea o plano simple.
# ==========================
# Cargar dataset
# ==========================

df = pd.read_csv("features.csv")

X = df.drop(columns=["genre", "file"])
y = df["genre"]

# ==========================
# Separar entrenamiento y prueba
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Normalizar los datos
# ==========================

# SVM es sensible a la escala de las variables, por lo que
# se normalizan todas las características.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Crear el modelo
# ==========================

modelo = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    random_state=42
)

# ==========================
# Entrenamiento
# ==========================

modelo.fit(X_train, y_train)

# ==========================
# Predicción
# ==========================

y_pred = modelo.predict(X_test)

# ==========================
# Resultados
# ==========================

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Matriz de confusión
# ==========================

ConfusionMatrixDisplay.from_estimator(
    modelo,
    X_test,
    y_test,
    xticks_rotation=45
)

plt.tight_layout()
plt.show()