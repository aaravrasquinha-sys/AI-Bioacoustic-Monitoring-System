import os
import librosa
import soundfile as sf
import numpy as np

# Settings
INPUT_DIR = "bird_calls"
OUTPUT_DIR = "processed_data"
SEGMENT_LEN = 3 
SR = 22050       
THRESHOLD = 0.02  # If a slice is quieter than this, it gets deleted

def slice_audio():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for bird_folder in os.listdir(INPUT_DIR):
        bird_path = os.path.join(INPUT_DIR, bird_folder)
        if not os.path.isdir(bird_path): continue

        print(f"Slicing: {bird_folder}...")
        out_bird_path = os.path.join(OUTPUT_DIR, bird_folder)
        os.makedirs(out_bird_path, exist_ok=True)

        for file_name in os.listdir(bird_path):
            file_path = os.path.join(bird_path, file_name)
            try:
                audio, _ = librosa.load(file_path, sr=SR)
                samples_per_seg = SEGMENT_LEN * SR
                
                for i in range(0, int(len(audio) // samples_per_seg)):
                    start = i * samples_per_seg
                    end = start + samples_per_seg
                    segment = audio[start:end]
                    
                    # Check if the segment has actual sound (RMS energy)
                    rms = np.sqrt(np.mean(segment**2))
                    
                    if rms > THRESHOLD:
                        chunk_name = f"{os.path.splitext(file_name)[0]}_chunk{i}.wav"
                        sf.write(os.path.join(out_bird_path, chunk_name), segment, SR)
            except Exception as e:
                print(f"Error on {file_name}: {e}")

if __name__ == "__main__":
    slice_audio()
    print("Done! Check your 'processed_data' folder.")