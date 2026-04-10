"""
TrafficCNN v2 — Convolutional Neural Network for Traffic Congestion Prediction
Architecture: 3-block Conv2D → GlobalAvgPool → Dense Head → Softmax(4)
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Using NumPy CNN simulation mode.")


# ─────────────────────────────────────────────
# NumPy-based CNN Simulation (no TF required)
# ─────────────────────────────────────────────

def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
def _relu(x): return np.maximum(0, x)
def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class NumPyCNNSimulator:
    """
    Lightweight NumPy simulation of the TrafficCNN.
    Produces realistic probability distributions based on input features.
    Used when TensorFlow is not available.
    """
    CONGESTION_LEVELS = {0:'Free Flow', 1:'Moderate', 2:'Heavy', 3:'Severe'}
    CONGESTION_COLORS = {0:'#00e676', 1:'#ffca28', 2:'#ff7043', 3:'#ff1744'}

    def __init__(self):
        np.random.seed(42)
        # Simulated learned weights (feature importances per class)
        self.W = np.array([
            # Free  Mod   Heavy  Severe  ← per class
            [ 0.9, -0.3, -0.5,  -0.8],  # speed (high=free)
            [-0.2,  0.4,  0.5,   0.7],  # volume
            [-0.3,  0.3,  0.6,   0.9],  # occupancy
            [-0.1,  0.1,  0.3,   0.5],  # incidents
            [ 0.4,  0.0, -0.2,  -0.4],  # weather (good=free)
            [ 0.1,  0.2, -0.1,  -0.2],  # time_sin
            [ 0.1,  0.1,  0.0,   0.0],  # time_cos
            [-0.2,  0.3,  0.5,   0.7],  # capacity_util
            [ 0.1,  0.0, -0.1,  -0.1],  # temperature
            [ 0.2,  0.0, -0.1,  -0.2],  # visibility
        ])
        self.bias = np.array([0.5, 0.3, 0.1, -0.1])

    def _conv_pool_sim(self, x):
        """Simulate Conv2D feature extraction using mean + std statistics"""
        mean_feat = x.mean(axis=0)  # (10,)
        std_feat = x.std(axis=0)    # (10,)
        max_feat = x.max(axis=0)    # (10,)
        # Combine: last timestep weighted more
        last = x[-1]
        combined = 0.4 * last + 0.3 * mean_feat + 0.2 * max_feat + 0.1 * std_feat
        return combined

    def predict_single(self, sample):
        """
        sample: (24, 10, 3) → mean channels → simulate forward pass
        Returns dict with prediction details
        """
        x = sample.mean(axis=-1)          # (24, 10)
        features = self._conv_pool_sim(x) # (10,)
        logits = features @ self.W + self.bias  # (4,)
        # Add noise for realism
        logits += np.random.normal(0, 0.1, 4)
        probs = _softmax(logits)

        class_idx = int(np.argmax(probs))
        return {
            "congestion_level": class_idx,
            "congestion_label": self.CONGESTION_LEVELS[class_idx],
            "congestion_color": self.CONGESTION_COLORS[class_idx],
            "confidence": float(probs[class_idx]),
            "probabilities": {
                self.CONGESTION_LEVELS[i]: float(probs[i])
                for i in range(4)
            }
        }


# ─────────────────────────────────────────────
# TensorFlow CNN Model
# ─────────────────────────────────────────────

class TrafficCNNModel:
    """
    3-Block Convolutional Neural Network for Traffic Congestion Prediction.

    Input:  (batch, 24, 10, 3)  ← 24h window, 10 features, 3 channels
    Output: (batch, 4)          ← softmax over [Free, Moderate, Heavy, Severe]

    Block 1: Conv2D(32) × 2 + BN + MaxPool(2,2) + Dropout(0.25)
    Block 2: Conv2D(64) × 2 + BN + MaxPool(2,2) + Dropout(0.25)
    Block 3: Conv2D(128) × 2 + BN + GlobalAvgPool + Dropout(0.4)
    Head:    Dense(256, ReLU) → Dense(128, ReLU) → Dense(4, Softmax)
    """

    LEVELS = {0:'Free Flow', 1:'Moderate', 2:'Heavy', 3:'Severe'}
    COLORS = {0:'#00e676', 1:'#ffca28', 2:'#ff7043', 3:'#ff1744'}

    def __init__(self, input_shape=(24, 10, 3), num_classes=4):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required for TrafficCNNModel. Use NumPyCNNSimulator instead.")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build()
        self._print_summary()

    def _conv_block(self, x, filters, kernel=(3,3), pool=True, drop=0.25):
        x = layers.Conv2D(filters, kernel, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if pool:
            x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(drop)(x)
        return x

    def _build(self):
        inp = keras.Input(shape=self.input_shape, name='traffic_24h_input')

        # Block 1 — Local patterns
        x = self._conv_block(inp, filters=32, pool=True, drop=0.25)

        # Block 2 — Mid-level patterns
        x = self._conv_block(x, filters=64, pool=True, drop=0.25)

        # Block 3 — Global patterns (no pool before GAP)
        x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (1,1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)

        # Classification head
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(self.num_classes, activation='softmax', name='congestion_softmax')(x)

        model = Model(inp, out, name='TrafficCNN_v2')
        model.compile(
            optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _print_summary(self):
        total = self.model.count_params()
        print(f"TrafficCNN v2 | Params: {total:,} | Input: {self.input_shape} | Classes: {self.num_classes}")

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        callbacks = [
            EarlyStopping('val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau('val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            ModelCheckpoint('models/best_cnn.h5', save_best_only=True, verbose=0),
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1
        )
        return history

    def predict_batch(self, X):
        probs = self.model.predict(X, verbose=0)
        idx = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        return idx, conf, probs

    def predict_single(self, sample):
        X = sample.reshape(1, *self.input_shape)
        probs = self.model.predict(X, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            "congestion_level": idx,
            "congestion_label": self.LEVELS[idx],
            "congestion_color": self.COLORS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {self.LEVELS[i]: float(probs[i]) for i in range(4)}
        }

    def save(self, path='models/traffic_cnn_v2.h5'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"✅ Model saved → {path}")

    def load(self, path='models/traffic_cnn_v2.h5'):
        self.model = keras.models.load_model(path)
        print(f"✅ Model loaded ← {path}")

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import classification_report, confusion_matrix
        idx, conf, probs = self.predict_batch(X_test)
        acc = np.mean(idx == y_test)
        print(f"\n{'='*50}")
        print(f"TrafficCNN v2 — Evaluation Results")
        print(f"{'='*50}")
        print(f"Overall Accuracy : {acc*100:.2f}%")
        print(f"Avg Confidence   : {conf.mean()*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, idx, target_names=list(self.LEVELS.values())))
        return acc, conf.mean()


def get_model(use_numpy_fallback=True):
    """Factory: returns TF model if available, else NumPy simulator"""
    if TF_AVAILABLE:
        return TrafficCNNModel()
    elif use_numpy_fallback:
        print("Using NumPy CNN simulator (TF not installed)")
        return NumPyCNNSimulator()
    else:
        raise RuntimeError("TensorFlow not available")
