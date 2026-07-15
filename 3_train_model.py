import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# --- 1. GPU INITIALIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Ensure WSL can see the Windows GPU drivers
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 RTX 5050 (Blackwell) Activated!")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU not found! Check your WSL NVIDIA drivers.")

# --- 2. UPDATED SETTINGS ---
# Using a relative path since you moved files into 'bird_brain'
DATA_DIR = "spectrogram_images" 
IMG_SIZE = (128, 128)
BATCH_SIZE = 64 
EPOCHS = 150

# --- 3. DATASET LOADING ---
if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Folder '{DATA_DIR}' not found in {os.getcwd()}")
    exit()

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2, 
    subset="training", 
    seed=42,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    label_mode='int'
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2, 
    subset="validation", 
    seed=42,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    label_mode='int'
)

class_names = raw_train_ds.class_names
num_classes = len(class_names)
print(f"✅ Training for {num_classes} classes: {class_names}")

# RTX 5050 Performance Optimization
train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = raw_val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 4. AGGRESSIVE ARCHITECTURE ---
data_augmentation = tf.keras.Sequential([
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.GaussianNoise(0.1), # Vital for your Lenovo mic's background hiss
])

base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True # Fully unfreeze for maximum specialization

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    
    # Deep Head for complex audio patterns
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-5), # Low rate for stable fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. CALLBACKS ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)

# --- 6. TRAINING ---
print("\n🔥 Commencing High-Performance Blackwell Training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_scheduler]
)

model.save("bird_monitor_v4_aggressive.h5")
print("\n🏆 Training Complete! Model saved as bird_monitor_v3_aggressive.h5")