import os
import sys
import numpy as np
import tensorflow as tf

# --- 1. GPU CONFIG ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- 2. SETTINGS ---
MODEL_PATH = "bird_monitor_v4_aggressive.h5"

# --- CORRECTED LINUX PATH FOR WSL ---
IMAGE_TO_TEST = "/home/aarav21/bird_brain/spectrogram_images/White-throated_Kingfisher/XC980502 - White-throated Kingfisher - Halcyon smyrnensis_chunk2.png"
DATA_DIR = "/home/aarav21/bird_brain/spectrogram_images"

def run_prediction():
    # A. Pre-flight Checks
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"❌ Error: Image not found at Linux path:\n{IMAGE_TO_TEST}")
        return

    # B. Load Species Names
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    # C. Load Model
    print("🧠 Loading AI model into RTX 5050...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # D. Prepare the Image
    img = tf.keras.utils.load_img(IMAGE_TO_TEST, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    
    # Check if model already includes a Rescaling layer (1./255)
    # If it does, we don't divide here. If it doesn't, we do.
    # Since our 'v3_aggressive' script HAD a Rescaling layer, we pass raw pixels.
    # If the confidence is weirdly low, uncomment the next line:
    # img_array = img_array / 255.0 
    
    img_array = tf.expand_dims(img_array, 0)

    # E. Predict
    print(f"📡 Analyzing: {os.path.basename(IMAGE_TO_TEST)}")
    predictions = model.predict(img_array, verbose=0)
    
    score = predictions[0]
    result_index = np.argmax(score)
    bird_name = class_names[result_index]
    confidence = 100 * score[result_index]

    # F. Display Result
    print("\n" + "═"*40)
    print(f"   🐦 AI BIOACOUSTIC RESULT")
    print("═"*40)
    
    # Formatting the name for display
    display_name = bird_name.replace("_", " ")

    if bird_name == "0_Background":
        print(f"   MATCH:      Background / Noise")
    elif confidence < 35:
        print(f"   MATCH:      Unknown (Low Confidence)")
        print(f"   CLOSEST:    {display_name}")
    else:
        print(f"   SPECIES:    {display_name}")
        
    print(f"   CONFIDENCE: {confidence:.2f}%")
    print("═"*40 + "\n")

if __name__ == "__main__":
    run_prediction()