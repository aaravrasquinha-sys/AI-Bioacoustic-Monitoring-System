import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = "processed_data"      # Where your 3-second slices are
OUTPUT_DIR = "spectrogram_images" # Where the pictures will go
SR = 22050                        # Standard sample rate
N_MELS = 128                      # Height of the image (Frequency bins)

def generate_spectrograms():
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Loop through each bird species folder
    for bird_folder in os.listdir(INPUT_DIR):
        bird_path = os.path.join(INPUT_DIR, bird_folder)
        if not os.path.isdir(bird_path):
            continue

        print(f"--- Converting {bird_folder} ---")
        out_bird_path = os.path.join(OUTPUT_DIR, bird_folder)
        os.makedirs(out_bird_path, exist_ok=True)

        for file_name in os.listdir(bird_path):
            if not file_name.endswith('.wav'):
                continue
                
            file_path = os.path.join(bird_path, file_name)
            
            try:
                # 1. Load audio
                y, sr = librosa.load(file_path, sr=SR)

                # 2. Create Mel-Spectrogram
                # n_mels=128 gives the AI enough "vertical" detail to see pitch
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
                
                # 3. Convert power to decibels (log scale)
                # This makes the bird call stand out from the background noise
                S_db = librosa.power_to_db(S, ref=np.max)

                # 4. Save as a clean image
                # 'magma' or 'viridis' are great because they show heat/intensity well
                img_name = f"{os.path.splitext(file_name)[0]}.png"
                img_path = os.path.join(out_bird_path, img_name)
                
                # Use plt.imsave to avoid axes, labels, and margins
                plt.imsave(img_path, S_db, cmap='magma', origin='lower')

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    generate_spectrograms()
    print("\nProcessing complete! Check the 'spectrogram_images' folder.")