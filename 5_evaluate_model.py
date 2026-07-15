import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. GPU CONFIG (same as training) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- 2. SETTINGS — must match 3_train_model.py exactly, or the val split won't match ---
MODEL_PATH = "bird_monitor_v4_aggressive.h5"
DATA_DIR = "spectrogram_images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
VAL_SPLIT = 0.2
SEED = 42

OUTPUT_DIR = "evaluation_results"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data folder not found: {DATA_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 3. Rebuild the SAME validation split used during training ---
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False  # keep order stable so predictions line up with labels
    )

    class_names = val_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Loaded validation set — {num_classes} classes: {class_names}")

    val_ds_perf = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 4. Load model ---
    print("🧠 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # --- 5. Overall loss / accuracy ---
    loss, accuracy = model.evaluate(val_ds_perf, verbose=0)
    print(f"\n📊 Validation Loss: {loss:.4f}")
    print(f"📊 Validation Accuracy: {accuracy*100:.2f}%")

    # --- 6. Collect predictions for detailed metrics ---
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- 7. Per-class precision / recall / F1 ---
    # Create an array of all possible class indices (0 to num_classes - 1)
    all_labels = np.arange(num_classes)

    report = classification_report(
        y_true, 
        y_pred, 
        labels=all_labels,        # Map predictions to all 12 classes explicitly
        target_names=class_names, 
        digits=3, 
        zero_division=0
    )
    print("\n📋 Classification Report:\n")
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Validation Loss: {loss:.4f}\n")
        f.write(f"Validation Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)

    # --- 8. Confusion matrix ---
    # Force the confusion matrix to structure itself as a complete 12x12 grid
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap='magma')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Validation Set")

    # annotate cell counts
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] < thresh else "black", fontsize=7)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\n🖼️  Confusion matrix saved to {cm_path}")

    # --- 9. Save a ready-to-paste markdown snippet for the README ---
    md_table = (
        "| Metric | Value |\n"
        "|---|---|\n"
        "| Validation accuracy | {accuracy*100:.1f}% |\n"
        "| Validation loss | {loss:.2f} |\n"
        "| Classes | {num_classes} |\n"
    )
    with open(os.path.join(OUTPUT_DIR, "results_table.md"), "w") as f:
        f.write(md_table)

    # --- 10. Also save raw numbers as JSON, useful for later charting ---
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "val_loss": float(loss),
            "val_accuracy": float(accuracy),
            "num_classes": num_classes,
            "class_names": class_names,
        }, f, indent=2)

    print(f"\n✅ Done! Everything saved in '{OUTPUT_DIR}/':")
    print("   - classification_report.txt")
    print("   - confusion_matrix.png")
    print("   - results_table.md   <- paste this straight into your README")
    print("   - metrics.json")


if __name__ == "__main__":
    main()