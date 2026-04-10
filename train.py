"""
TrafficCNN v2 — Training Script
Trains, evaluates, and saves the CNN model.
Run: python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

def main():
    print("=" * 60)
    print("  TrafficCNN v2 — Training Pipeline")
    print("=" * 60)

    from utils.data_generator import TrafficDataGenerator
    gen = TrafficDataGenerator()
    X_tr, X_val, y_tr, y_val = gen.split(n_samples=6000)
    print(f"\nInput shape  : {X_tr.shape}")
    print(f"Class dist   : {np.bincount(y_tr)} (Free/Moderate/Heavy/Severe)")

    try:
        import tensorflow as tf
        from models.cnn_model import TrafficCNNModel
        print(f"TF version   : {tf.__version__}")

        model = TrafficCNNModel(input_shape=(24,10,3), num_classes=4)
        history = model.train(X_tr, y_tr, X_val, y_val, epochs=50, batch_size=64)
        acc, conf = model.evaluate(X_val, y_val)
        model.save('models/traffic_cnn_v2.h5')

        # Plot
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor('#06080f')
            for ax in (ax1, ax2):
                ax.set_facecolor('#111520')
                for sp in ax.spines.values(): sp.set_color('#1e2d45')
                ax.tick_params(colors='#5a6478')
            ax1.plot(history.history['accuracy'],   color='#00e5ff', label='Train', lw=2)
            ax1.plot(history.history['val_accuracy'],color='#7c4dff', label='Val',   lw=2)
            ax1.set_title('Accuracy', color='white', fontsize=12)
            ax1.legend(facecolor='#161b28', labelcolor='white')
            ax2.plot(history.history['loss'],    color='#ff7043', label='Train', lw=2)
            ax2.plot(history.history['val_loss'],color='#ff1744', label='Val',   lw=2)
            ax2.set_title('Loss', color='white', fontsize=12)
            ax2.legend(facecolor='#161b28', labelcolor='white')
            plt.tight_layout()
            plt.savefig('models/training_curves.png', dpi=140, facecolor='#06080f', bbox_inches='tight')
            print("📊 Training curves saved → models/training_curves.png")
        except Exception as e:
            print(f"Plot skipped: {e}")

    except ImportError:
        print("⚠️  TensorFlow not installed. Using NumPy CNN simulator.")
        from models.cnn_model import NumPyCNNSimulator
        sim = NumPyCNNSimulator()
        sample = gen.realtime_sample()
        r = sim.predict_single(sample)
        print(f"\nDemo inference → {r['congestion_label']} ({r['confidence']*100:.1f}%)")

    print("\n✅ Done! Run: python app.py")

if __name__ == '__main__':
    main()
