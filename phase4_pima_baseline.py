import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement Pima (URL directe, pas de compte Kaggle requis)
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(pima_url, names=cols)

# TODO : afficher df['Outcome'].value_counts() pour voir la distribution de classes
# noter les proportions exactes avant d'entraîner quoi que ce soit
print(df['Outcome'].value_counts())

# TODO : afficher (df == 0).sum() pour toutes les colonnes
# Glucose=0, BMI=0, Insulin=0, SkinThickness=0 sont physiologiquement impossibles
# ce sont des NaN déguisés en zéros, encodage courant dans les datasets médicaux anciens
# on les laisse pour l'instant, mais les noter : c'est un point de fragilité réel
print((df == 0).sum()) 
 
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# TODO : split train/test 80/20, random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO : instancier StandardScaler, le fitter sur X_train UNIQUEMENT, transformer train et test
# (fitter sur X_test = data leakage : le modèle verrait les stats du test avant évaluation)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# TODO : construire un modèle Sequential binaire
# architecture : Dense(64, relu, input_shape=(8,)) -> Dense(32, relu) -> Dense(1, sigmoid)
# compiler avec optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# TODO : entraîner 100 epochs, validation_split=0.2, batch_size=32
# stocker le résultat dans une variable `history`
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2, 
    verbose=1
)

# TODO : afficher la val_accuracy finale (max sur toutes les epochs)
# et vérifier que model.predict(X_val).mean() est proche de 0.35 (pas 0.05)
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {final_val_accuracy:.4f}")