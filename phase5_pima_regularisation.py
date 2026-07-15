import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase4_pima_baseline import X_train_norm, y_train

# X_train_norm, X_val, y_train, y_val issus de la Phase 4
# (ou reconstruire le preprocessing en tête de fichier)
def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    """
    Modèle Pima avec régularisation L2 optionnelle et Dropout optionnel.
    Si use_dropout=True, insère un Dropout(0.3) après chaque couche cachée.
    """
    # TODO : créer un keras.Sequential vide
    model = keras.Sequential()
    # TODO : ajouter Dense(64, relu, input_shape=(8,)) avec kernel_regularizer=regularizers.l2(l2_lambda)
    model.add(layers.Dense(64, activation='relu', input_shape=(8,), kernel_regularizer=regularizers.l2(l2_lambda)))
    # TODO : si use_dropout est True, ajouter layers.Dropout(0.3) immédiatement après
    if use_dropout:
        model.add(layers.Dropout(0.3))
    # TODO : ajouter Dense(32, relu) avec kernel_regularizer=regularizers.l2(l2_lambda)
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    # TODO : si use_dropout est True, ajouter layers.Dropout(0.3) immédiatement après
    if use_dropout:
        model.add(layers.Dropout(0.3))
    # TODO : ajouter Dense(1, activation='sigmoid') en couche de sortie (pas de régularisation ici)
    model.add(layers.Dense(1, activation='sigmoid'))
    # TODO : compiler avec optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # TODO : retourner le modèle
    return model

# Callback Early Stopping : surveille val_loss, arrêt si pas d'amélioration pendant 15 epochs.
# restore_best_weights=True récupère les poids du meilleur epoch, pas du dernier.
early_stopping = keras.callbacks.EarlyStopping(
monitor='val_loss', patience=15, restore_best_weights=True
)
# --- Configuration 1 : baseline non régularisé (architecture identique à Phase 4) ---
print("Entraînement 1/3 : Baseline")
# TODO : construire model_baseline avec build_pima_regularized(l2_lambda=0.0, use_dropout=False)
model_baseline = build_pima_regularized(l2_lambda=0.0, use_dropout=False)

# TODO : appeler model_baseline.fit(X_train_norm, y_train, epochs=300, validation_split=0.2,
# callbacks=[early_stopping], verbose=0)
model_baseline = build_pima_regularized(l2_lambda=0.0, use_dropout=False)

# TODO : stocker le retour dans history_baseline
history_baseline = model_baseline.fit(
    X_train_norm, y_train, epochs=300, validation_split=0.2, 
    callbacks=[early_stopping], verbose=0
)

# TODO : afficher l'epoch d'arrêt (len(history_baseline.history['val_loss'])) et max(val_accuracy)
print(f"Epoch d'arrêt (Baseline) : {len(history_baseline.history['val_loss'])}")
print(f"Max val_accuracy (Baseline) : {max(history_baseline.history['val_accuracy'])}")

# --- Configuration 2 : L2 seul (l2_lambda=0.01, use_dropout=False) ---
print("Entraînement 2/3 : L2 seul")
# TODO : construire model_l2 avec build_pima_regularized(l2_lambda=0.01, use_dropout=False)
model_l2 = build_pima_regularized(l2_lambda=0.01, use_dropout=False)

# TODO : entraîner avec les mêmes paramètres et le même callback
history_l2 = model_l2.fit(
    X_train_norm, y_train, epochs=300, validation_split=0.2, 
    callbacks=[early_stopping], verbose=0
)
 
# TODO : stocker dans history_l2, afficher l'epoch d'arrêt et le max val_accuracy
epoch_arret_l2 = len(history_l2.history['val_loss'])
max_acc_l2 = max(history_l2.history['val_accuracy'])
print(f"L2 stoppé à l'epoch {epoch_arret_l2}")
print(f"Max val_accuracy (L2) : {max_acc_l2:.4f}")


# --- Configuration 3 : L2 + Dropout (l2_lambda=0.01, use_dropout=True) ---
print("Entraînement 3/3 : L2 + Dropout")
# TODO : construire model_l2_drop avec build_pima_regularized(l2_lambda=0.01, use_dropout=True)
model_l2_drop = build_pima_regularized(l2_lambda=0.01, use_dropout=True)
# TODO : entraîner et stocker dans history_l2_drop
history_l2_drop = model_l2_drop.fit(
    X_train_norm, y_train, epochs=300, validation_split=0.2, 
    callbacks=[early_stopping], verbose=0
)
# TODO : afficher l'epoch d'arrêt et le max val_accuracy
epoch_arret_l2_drop = len(history_l2_drop.history['val_loss'])
max_acc_l2_drop = max(history_l2_drop.history['val_accuracy'])
print(f"L2 + Dropout stoppé à l'epoch {epoch_arret_l2_drop}")
print(f"Max val_accuracy (L2 + Dropout) : {max_acc_l2_drop:.4f}")

# TODO : tracer les trois courbes val_loss côte à côte avec matplotlib
# titres : "Baseline", "L2 seul", "L2 + Dropout"
# ajouter une ligne verticale à l'epoch d'arrêt pour chaque config
# sauvegarder en phase5_pima_3configs.png
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

histories = [history_baseline, history_l2, history_l2_drop]
titles = ["Baseline (Pas de reg)", "L2 seul", "L2 + Dropout"]

for i in range(3):
    axes[i].plot(histories[i].history['val_loss'], label='val_loss')
    axes[i].axvline(x=len(histories[i].history['val_loss'])-1, color='r', linestyle='--', label='Early Stopping')
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Validation Loss')
    axes[i].legend()

plt.suptitle("Impact de la régularisation et du Early Stopping sur Pima Diabetes", fontsize=14)
plt.savefig("phase5_pima_3configs.png", dpi=100, bbox_inches='tight')
