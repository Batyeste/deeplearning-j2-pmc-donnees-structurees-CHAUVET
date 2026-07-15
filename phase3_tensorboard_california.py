import datetime
from tensorflow import keras

from phase1_pipeline_california import X_train_norm, X_val_norm, X_train, X_val, y_train, y_val
from phase2_baseline_regression import build_regression_model

def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=100):
    """Entraîne un modèle de régression avec un callback TensorBoard horodaté."""
    # TODO : construire log_dir = f"logs/fit/{run_name}_" + timestamp au format HHMMSS
    # (utiliser datetime.datetime.now().strftime("%H%M%S"))
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_{timestamp}"

    # TODO : instancier keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # histogram_freq=1 : enregistrer les distributions des poids à chaque epoch
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # TODO : instancier le modèle via build_regression_model(input_dim=8)
    model = build_regression_model(input_dim=8)

    # TODO : appeler model.fit avec callbacks=[tb_callback],
    # validation_data=(X_val, y_val), epochs=epochs, batch_size=32, verbose=0
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[tb_callback]
    )

    # TODO : afficher un message confirmant le run et son chemin de logs
    # ex : f"Run '{run_name}' terminé. Logs dans {log_dir}"
    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")
    # TODO : retourner (model, history)
    return model, history

# Run 1 : données normalisées (bon comportement attendu)
print("Lancement Run 1")
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val,
    run_name="california_norm"
)
# Run 2 : données brutes (comportement dégradé à observer)
print("Lancement Run 2 (brut)")
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val,
    run_name="california_raw"
)
# Lancer TensorBoard dans un terminal séparé :
# tensorboard --logdir=logs/fit
# Puis ouvrir http://localhost:6006