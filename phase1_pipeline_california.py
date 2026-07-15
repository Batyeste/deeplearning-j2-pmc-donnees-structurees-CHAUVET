import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger le dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# StandardScaler (https://scikitlearn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# normalise chaque feature : (X - mean) / std.
# Résultat : mean = 0, std = 1 sur le train set.
# Pourquoi fitter sur X_train uniquement : si on fitte sur X entier,
# les stats du scaler "voient" le test set avant l'évaluation (data leakage).

# TODO : faire un premier split train/test avec test_size=0.2 et random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# TODO : faire un second split train/val sur le résultat précédent (val_size=0.2 du train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# TODO : instancier un StandardScaler et le fitter sur X_train UNIQUEMENT
scaler = StandardScaler()
scaler.fit(X_train)

# TODO : transformer X_train, X_val, X_test avec le scaler fitté
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# TODO : afficher les shapes de X_train, X_val, X_test
print("Shapes:")
print("X_train:", X_train_norm.shape)
print("X_val:", X_val_norm.shape)
print("X_test:", X_test_norm.shape)

# TODO : afficher les stats descriptives de X_train_norm (mean et std par feature)
print("\nStats descriptives de X_train_norm :")
print("Mean par feature:", np.mean(X_train_norm, axis=0))
print("Std par feature:", np.std(X_train_norm, axis=0))
# TODO : afficher les feature_names du dataset ET vérifier qu'il y en a bien 8
print("\nFeature names:")
for i, name in enumerate(housing.feature_names):
    print(f"{name}: {i}")
print(f"Nombre de features: {len(housing.feature_names)}")
