#   Copyright 2025 BirdNET-Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import Model

from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESP_IDF_v5_2
from biodcase_tiny.feature_extraction.feature_extraction import make_constants
from config import Config, load_config
from paths import KERAS_MODEL_PATH, REFERENCE_DATASET_PATH, GEN_CODE_DIR, TFLITE_MODEL_PATH


def compute_mel_range(reference_dataset: tf.data.Dataset) -> tuple[int, int]:
    """Computes the global min/max of the training-time float log-mel features
    across the reference/calibration dataset.

    These bounds get baked into the on-device FeatureConfig and used by
    feature_extraction.cpp's rescale_to_int8() to map live log-mel values into
    int8. Previously these were never set (defaulted to 0/0), which made
    data_range collapse to zero and pinned every on-device feature at -128
    regardless of input -- i.e. all live inference was reading dead features.

    The dataset yields already-computed float spectrograms (the same values
    process_window(..., inference_mode=False) produces), matching what
    _create_model_buf()'s representative_dataset_gen() feeds the TFLite
    converter for quantization calibration -- so this reuses exactly the data
    the model itself was calibrated against, rather than a separate estimate.
    """
    mel_min = np.inf
    mel_max = -np.inf
    for example_spectrograms, _ in reference_dataset:
        arr = example_spectrograms.numpy() if hasattr(example_spectrograms, "numpy") else np.asarray(example_spectrograms)
        if arr.size == 0:
            continue
        mel_min = min(mel_min, float(arr.min()))
        mel_max = max(mel_max, float(arr.max()))

    if not np.isfinite(mel_min) or not np.isfinite(mel_max):
        raise ValueError(
            "compute_mel_range() found no data in reference_dataset -- cannot "
            "compute mel_range_min/max. On-device features would silently "
            "collapse to a constant value without these."
        )
    if mel_max <= mel_min:
        raise ValueError(
            f"reference_dataset mel range is degenerate (min={mel_min}, "
            f"max={mel_max}). Check that reference_dataset actually contains "
            "varied audio, not a single silent/constant clip."
        )

    return int(np.floor(mel_min)), int(np.ceil(mel_max))


def create_target(
    model_path: Path,
    reference_dataset_path: Path | None,
    config: Config,
    quantize: bool = False,
):
    if model_path.suffix == ".keras":
        model = keras.models.load_model(model_path)
        reference_dataset = tf.data.Dataset.load(str(reference_dataset_path))
    elif model_path.suffix == ".tflite":
        with model_path.open("rb") as f:
            model = f.read()
        reference_dataset = None
    else:
        raise ValueError("Only Keras and tflite format supported")

    if reference_dataset is not None:
        mel_range_min, mel_range_max = compute_mel_range(reference_dataset)
        print(f"Computed mel_range_min={mel_range_min}, mel_range_max={mel_range_max} "
              f"from reference dataset.")
    else:
        # No reference dataset available (loading an already-exported .tflite
        # directly). We can't derive the calibration range in this path --
        # fall back to 0/1 rather than 0/0 so rescale_to_int8's data_range is
        # at least non-zero, but this will NOT produce correct features on
        # device. Prefer going through the .keras path with a reference
        # dataset whenever regenerating firmware.
        print("WARNING: no reference_dataset available, cannot compute real "
              "mel_range_min/max. On-device features will not be scaled "
              "correctly. Re-run from the .keras model with a reference "
              "dataset to fix this properly.")
        mel_range_min, mel_range_max = 0, 1

    dp_c = config.data_preprocessing
    fe_c = config.feature_extraction
    feature_config = make_constants(
        sample_rate=dp_c.sample_rate,
        win_samples=fe_c.window_len, window_scaling_bits=fe_c.window_scaling_bits,
        mel_n_channels=fe_c.mel_n_channels, mel_low_hz=fe_c.mel_low_hz, mel_high_hz=fe_c.mel_high_hz,
        mel_post_scaling_bits=fe_c.mel_post_scaling_bits,
        mel_range_min=mel_range_min, mel_range_max=mel_range_max,
    )
    target = ESPTarget(model, feature_config, reference_dataset, quantize=quantize)
    target.validate()
    return target


def generate_and_flash(config: Config, target: ESPTarget, gen_code_dir: Path):
    toolchain = ESP_IDF_v5_2(config.embedded_code_generation.serial_device)
    src_path = gen_code_dir / "src"
    src_path.mkdir(exist_ok=True)

    target.process_target_templates(src_path)
    toolchain.compile(src_path=src_path)
    toolchain.flash(src_path=src_path)
    toolchain.monitor(src_path=src_path)


def run_embedded_code_generation(
    config: Config,
    model_path: Path = KERAS_MODEL_PATH,
    reference_dataset_path: Path = REFERENCE_DATASET_PATH,
    tflite_model_path: Path = TFLITE_MODEL_PATH,
    gen_code_dir: Path = GEN_CODE_DIR,
    quantize: bool = True,
):
    target = create_target(model_path, reference_dataset_path, config, quantize)
    tflite_model_buf = target.get_model_buf()
    with tflite_model_path.open("wb") as f:
        f.write(tflite_model_buf)
    generate_and_flash(config, target, gen_code_dir)


if __name__ == '__main__':
    config = load_config()
    run_embedded_code_generation(config)