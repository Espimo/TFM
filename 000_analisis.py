import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
import os
import json
from datetime import datetime

# Configuración inicial
batch_size = 16
epochs = 10  # Para depuración rápida, aumentarlo más tarde
learning_rate = 3e-5
output_dir = 'job_classification_model_custom'
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Configuración del archivo de log
log_file = os.path.join(output_dir, 'training_and_evaluation_log.txt')
os.makedirs(output_dir, exist_ok=True)

def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

# Configuración de la GPU
def setup_gpu():
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            log_message("GPU configured for dynamic memory growth")
        except RuntimeError as e:
            log_message(f"Error configuring GPU: {e}")
    else:
        log_message("No GPUs detected. Training will proceed on CPU")

def load_and_preprocess_data(min_samples_per_class=100):
    log_message("Loading embeddings and labels...")

    # Cargar embeddings
    job_summary_embeddings = np.load('job_summary_embeddings.npy')

    # Cargar y filtrar etiquetas
    labels_df = pd.read_csv('dataset_skills_binarizado.csv')
    class_counts = labels_df.sum()
    valid_columns = class_counts[class_counts >= min_samples_per_class].index
    labels_df = labels_df[valid_columns]
    labels = labels_df.values

    assert job_summary_embeddings.shape[0] == labels.shape[0], "Mismatch in data dimensions"

    X_train, X_test, y_train, y_test = train_test_split(
        job_summary_embeddings,
        labels,
        test_size=0.2,
        random_state=42
    )

    log_message(f"Data split completed:")
    log_message(f"- Training samples: {X_train.shape[0]}")
    log_message(f"- Test samples: {X_test.shape[0]}")
    log_message(f"- Features dimension: {X_train.shape[1]}")
    log_message(f"- Labels dimension: {y_train.shape[1]}")

    return X_train, X_test, y_train, y_test, valid_columns.tolist()

def create_tf_dataset(embeddings, labels, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
    if is_training:
        dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def compute_metrics(y_true, y_pred):
    metrics = {
        'hamming_loss': float(hamming_loss(y_true, y_pred)),
        'exact_match_ratio': float(accuracy_score(y_true, y_pred)),
        'f1_score_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1_score_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }
    return metrics

def build_custom_model(input_shape, num_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_labels, activation='sigmoid')
    ])
    return model

def train_and_evaluate():
    log_message("Starting training process...")

    X_train, X_test, y_train, y_test, label_list = load_and_preprocess_data(min_samples_per_class=100)
    num_labels = y_train.shape[1]

    log_message(f"Model configuration:")
    log_message(f"- Number of labels: {num_labels}")
    log_message(f"- Batch size: {batch_size}")
    log_message(f"- Epochs: {epochs}")
    log_message(f"- Learning rate: {learning_rate}")

    model = build_custom_model(X_train.shape[1], num_labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=1,
            verbose=1,
            mode='min'
        )
    ]

    history = model.fit(
        create_tf_dataset(X_train, y_train, is_training=True),
        validation_data=create_tf_dataset(X_test, y_test, is_training=False),
        epochs=epochs,
        callbacks=callbacks
    )

    model.save(output_dir)
    log_message("Model saved")

    log_message("Evaluating model on test data...")
    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    metrics_results = compute_metrics(y_test, y_pred)

    with open(f'{output_dir}/model_metrics.json', 'w') as f:
        json.dump(metrics_results, f)

    log_message("Evaluation metrics:")
    for metric, value in metrics_results.items():
        log_message(f"- {metric}: {value:.4f}")

    with open(f'{output_dir}/label_list.json', 'w') as f:
        json.dump(label_list, f)

    log_message(f"Model and results saved in '{output_dir}'")

def main():
    setup_gpu()
    train_and_evaluate()

if __name__ == "__main__":
    main()
