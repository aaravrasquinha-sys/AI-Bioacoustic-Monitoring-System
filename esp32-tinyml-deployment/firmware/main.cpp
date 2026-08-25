/* Copyright 2024 The TensorFlow Authors
   Copyright 2025 BirdNET team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This code was adapted from the benchmark utility of TFLM.

MODIFIED for live streaming inference on ESP32-S3 + INMP441:
  - raw-PCM ring buffer with sliding window (stride 512 over 4096-sample window)
  - (n_windows, n_mel) feature accumulation buffer, filled one row per window
  - real feature buffer copied into the interpreter input tensor (no random input)
  - saturating 24-bit -> 16-bit mic conversion with tunable gain
  - RMS / feature-range / probability instrumentation + detection debounce
==============================================================================*/

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <random>
#include <type_traits>
#include <span>
#include <cinttypes>

#include "esp_heap_caps.h"

#include "esp_micro_profiler.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "metrics.h"
#include "op_resolver.h"
#include "model.h"
#include "esp_micro_profiler.h"
#include "feature_config_generated.h"
#include "feature_config.h"
#include "feature_extraction.h"

#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"


namespace tflite {
namespace {

using Profiler = ::benchmark::MicroProfiler;

constexpr size_t kTensorArenaSize = 6000000;
uint8_t* tensor_arena;

// ---------------------------------------------------------------------------
// Streaming configuration
// ---------------------------------------------------------------------------

// MUST match pipeline_config.yaml feature_extraction.window_stride used during
// training (sliding_window_view stride). Window length comes from the feature
// config's hanning window size and must match window_len.
constexpr size_t kWindowStride = 512;

// Run the model once every kClassifyHop windows instead of every window.
// Budget check: one window arrives every kWindowStride/16000 s = 32 ms.
// Work per hop = kClassifyHop * t_features + t_invoke = 8*3.3 + 86 = 112 ms
// against a budget of 8*32 = 256 ms. Roughly 44% duty cycle.
constexpr size_t kClassifyHop = 8;

// INMP441 emits 24-bit data left-justified in a 32-bit frame, so (raw >> 8) is
// a signed 24-bit value. Shifting down by 8 more yields true 16-bit PCM.
// Lower this (6, 4, ...) to add 6 dB of gain per step if the printed RMS is too
// small; raise it if FEAT max saturates at +127. Result is always clamped, so
// over-gain saturates instead of wrapping.
constexpr int kMicGainShift = 8;

// Detection reporting. kTargetClass is the "call present" index.
constexpr float kDetectThreshold = 0.5f;
constexpr int kDetectConsecutive = 3;

// Set true to bypass the microphone and feed a deterministic synthetic tone.
// Use this to compare on-device feature rows against Python's process_window()
// for the same input buffer before trusting live audio.
constexpr bool kSelfTest = false;
constexpr float kSelfTestFreqHz = 4000.0f;   // in the yellowhammer band
constexpr int16_t kSelfTestAmplitude = 8000;

// ---------------------------------------------------------------------------
// CRC32 (kept for optional feature-buffer sanity checks)
// ---------------------------------------------------------------------------

constexpr uint32_t kCrctabLen = 256;
uint32_t crctab[kCrctabLen];

void GenCRC32Table() {
  constexpr uint32_t kPolyN = 0xEDB88320;
  for (size_t index = 0; index < kCrctabLen; index++) {
    crctab[index] = index;
    for (int i = 0; i < 8; i++) {
      if (crctab[index] & 1) {
        crctab[index] = (crctab[index] >> 1) ^ kPolyN;
      } else {
        crctab[index] >>= 1;
      }
    }
  }
}

uint32_t ComputeCRC32(const uint8_t* data, const size_t data_length) {
  uint32_t crc32 = ~0U;
  for (size_t i = 0; i < data_length; i++) {
    const uint32_t index = (crc32 ^ data[i]) & (kCrctabLen - 1);
    crc32 = (crc32 >> 8) ^ crctab[index];
  }
  crc32 ^= ~0U;
  return crc32;
}

// ---------------------------------------------------------------------------
// INMP441 I2S microphone capture
// Wiring: SD/DIN = GPIO2, SCK/BCLK = GPIO41, WS = GPIO42, L/R -> GND (left slot).
// Sample rate must match pipeline_config.yaml data_preprocessing.sample_rate.
// ---------------------------------------------------------------------------

constexpr gpio_num_t kI2sBckPin = GPIO_NUM_41;
constexpr gpio_num_t kI2sWsPin = GPIO_NUM_42;
constexpr gpio_num_t kI2sDinPin = GPIO_NUM_2;
constexpr uint32_t kI2sSampleRate = 16000;

i2s_chan_handle_t rx_handle = nullptr;

void InitI2sMic() {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  // The default DMA depth (~90 ms) is smaller than the ~112 ms of work done per
  // classify hop, during which no i2s_channel_read is issued. Enlarge it so the
  // driver keeps capturing across an Invoke() instead of dropping samples.
  // 32-bit mono = 4 bytes/frame, so dma_frame_num must stay under 1023.
  chan_cfg.dma_desc_num = 8;
  chan_cfg.dma_frame_num = 511;   // 8 * 511 = 4088 frames ~= 255 ms
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, nullptr, &rx_handle));

  i2s_std_config_t std_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kI2sSampleRate),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
          I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = kI2sBckPin,
          .ws = kI2sWsPin,
          .dout = I2S_GPIO_UNUSED,
          .din = kI2sDinPin,
          .invert_flags = {
              .mclk_inv = false,
              .bclk_inv = false,
              .ws_inv = false,
          },
      },
  };
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  // L/R tied to GND -> left slot

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

// Reads exactly n_samples frames and appends them as 16-bit PCM at dst.
//
// The INMP441 emits 24-bit data left-justified in the 32-bit frame, so
// (raw >> 8) is a signed 24-bit sample spanning +/-2^23. The previous code cast
// that straight to int16_t, which discarded the sign bit and WRAPPED on any
// sample above ~32768 (roughly 76 dB SPL) -- exactly the level a speaker held
// near the mic produces. Shift down to a true 16-bit range and clamp so loud
// input saturates gracefully instead of folding over.
void ReadPcmBlock(std::span<int32_t> raw_buf, int16_t* dst, size_t n_samples) {
  size_t bytes_read = 0;
  ESP_ERROR_CHECK(i2s_channel_read(
      rx_handle, raw_buf.data(), n_samples * sizeof(int32_t), &bytes_read,
      portMAX_DELAY));
  const size_t got = bytes_read / sizeof(int32_t);
  for (size_t i = 0; i < n_samples; ++i) {
    int32_t s = (i < got) ? (raw_buf[i] >> 8) : 0;   // signed 24-bit
    s >>= kMicGainShift;
    dst[i] = static_cast<int16_t>(std::clamp<int32_t>(s, INT16_MIN, INT16_MAX));
  }
}

// Deterministic stand-in for the microphone, used by the parity test.
void SynthPcmBlock(int16_t* dst, size_t n_samples, uint64_t sample_index) {
  constexpr float kTwoPi = 6.283185307179586f;
  for (size_t i = 0; i < n_samples; ++i) {
    const float t = static_cast<float>(sample_index + i) / kI2sSampleRate;
    dst[i] = static_cast<int16_t>(
        kSelfTestAmplitude * sinf(kTwoPi * kSelfTestFreqHz * t));
  }
}

uint32_t ComputeRms(const int16_t* p, size_t n) {
  uint64_t acc = 0;
  for (size_t i = 0; i < n; ++i) {
    const int32_t v = p[i];
    acc += static_cast<uint64_t>(v * v);
  }
  return static_cast<uint32_t>(sqrt(static_cast<double>(acc) / n));
}

// ---------------------------------------------------------------------------

[[nodiscard]] int Benchmark(const uint8_t* model_data,
                            const uint8_t* feature_extractor_data) {
  static Profiler profiler;
  TfLiteStatus status;

  tensor_arena = reinterpret_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));
  if (tensor_arena == nullptr) {
    MicroPrintf("Failed to allocate tensor arena");
    return -1;
  }

  const FeatureConfigs::FeatureConfig* fc =
      FeatureConfigs::GetFeatureConfig(feature_extractor_data);

  const size_t win_len = fc->hanning_window()->size();
  const size_t n_mel = fc->fb_config()->num_channels();

  MicroPrintf("Window length     : %u samples", (unsigned)win_len);
  MicroPrintf("Window stride     : %u samples", (unsigned)kWindowStride);
  MicroPrintf("Mel channels      : %u", (unsigned)n_mel);
  MicroPrintf("mel_range_min/max : %d / %d",
              (int)fc->mel_range_min(), (int)fc->mel_range_max());

  if (kWindowStride > win_len) {
    MicroPrintf("kWindowStride must not exceed the window length");
    return -1;
  }

  // Raw PCM ring: the most recent win_len samples, oldest first. A separate
  // buffer is required because extract_features() consumes its input in place
  // (Hanning, auto-scale and FFT all write back into in_audio).
  int16_t* pcm_win = (int16_t*)heap_caps_calloc(
      win_len, sizeof(int16_t), MALLOC_CAP_DEFAULT);

  // Complex-interleaved scratch handed to extract_features(); rebuilt from
  // pcm_win each step because the previous call destroyed it.
  int16_t* audio_win_arr = (int16_t*)heap_caps_aligned_alloc(
      16, win_len * sizeof(int16_t) * 2, MALLOC_CAP_DEFAULT);

  // Own allocation: previously this aliased audio_win, so feature-extraction
  // intermediates overwrote the audio still being read.
  uint8_t* scratch_arr = (uint8_t*)heap_caps_aligned_alloc(
      16, win_len * sizeof(int16_t) * 2, MALLOC_CAP_DEFAULT);

  int32_t* i2s_raw_arr = (int32_t*)heap_caps_malloc(
      kWindowStride * sizeof(int32_t), MALLOC_CAP_DEFAULT);

  if (!pcm_win || !audio_win_arr || !scratch_arr || !i2s_raw_arr) {
    MicroPrintf("Failed to allocate audio buffers");
    return -1;
  }

  std::span<int16_t> audio_win = {audio_win_arr, win_len * 2};
  std::span<uint8_t> scratch_buffer = {scratch_arr, win_len * sizeof(int16_t) * 2};
  std::span<int32_t> i2s_raw = {i2s_raw_arr, kWindowStride};

  init_feature_extraction(fc);
  if (!kSelfTest) {
    InitI2sMic();
  }

  const tflite::Model* model = tflite::GetModel(model_data);

  TflmOpResolver op_resolver;
  status = CreateOpResolver(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("tflite::CreateOpResolver failed");
    return -1;
  }

  tflite::RecordingMicroInterpreter interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);

  status = interpreter.AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("tflite::MicroInterpreter::AllocateTensors failed");
    return -1;
  }
  profiler.ClearEvents();
  interpreter.GetMicroAllocator().PrintAllocations();

  TfLiteTensor* model_input = interpreter.input_tensor(0);
  if (model_input->type != kTfLiteInt8) {
    MicroPrintf("Expected an int8 input tensor, got type %d. Re-export the "
                "model with quantize=True.", (int)model_input->type);
    return -1;
  }

  // Derive the window count from the tensor rather than a fixed dim index, so
  // this works for both (1, T, M) and (1, T, M, 1) input shapes.
  const size_t n_windows = model_input->bytes / n_mel;
  if (n_windows * n_mel != model_input->bytes) {
    MicroPrintf("Input tensor size %u is not a multiple of n_mel %u",
                (unsigned)model_input->bytes, (unsigned)n_mel);
    return -1;
  }

  MicroPrintf("Model input       : %u bytes -> %u windows x %u mel",
              (unsigned)model_input->bytes, (unsigned)n_windows, (unsigned)n_mel);
  MicroPrintf("Input quant       : scale=%.6f zero_point=%d",
              model_input->params.scale, (int)model_input->params.zero_point);
  MicroPrintf("Context           : %u samples (%.2f s)",
              (unsigned)(win_len + (n_windows - 1) * kWindowStride),
              (win_len + (n_windows - 1) * kWindowStride) /
                  (float)kI2sSampleRate);

  // Circular (n_windows, n_mel) accumulation buffer, one row per window.
  int8_t* features = (int8_t*)heap_caps_calloc(
      n_windows * n_mel, sizeof(int8_t), MALLOC_CAP_DEFAULT);
  if (features == nullptr) {
    MicroPrintf("Failed to allocate feature buffer");
    return -1;
  }

  TfLiteTensor* out_tensor = interpreter.output_tensor(0);
  const size_t n_out = out_tensor->bytes;
  const float out_scale = out_tensor->params.scale;
  const int out_zp = out_tensor->params.zero_point;
  const size_t target_class = (n_out > 1) ? 1 : 0;

  MicroPrintf("Model output      : %u classes, scale=%.6f zero_point=%d",
              (unsigned)n_out, out_scale, out_zp);
  MicroPrintf("");

  if (kSelfTest) {
    MicroPrintf("*** SELF TEST MODE: synthetic %.0f Hz tone, mic bypassed ***",
                kSelfTestFreqHz);
  }
  MicroPrintf("Priming %u windows (~%.2f s) before first inference...",
              (unsigned)n_windows,
              (win_len + (n_windows - 1) * kWindowStride) /
                  (float)kI2sSampleRate);
  MicroPrintf("");

  size_t head = 0;            // next row to write; once full, also the oldest
  size_t filled = 0;          // rows containing real data
  size_t since_classify = 0;
  int consecutive_hits = 0;
  uint64_t sample_index = 0;  // self-test phase continuity
  bool self_test_reported = false;

  while (true) {
    // --- slide the ring by one stride and append fresh samples -------------
    memmove(pcm_win, pcm_win + kWindowStride,
            (win_len - kWindowStride) * sizeof(int16_t));
    int16_t* tail = pcm_win + win_len - kWindowStride;
    if (kSelfTest) {
      SynthPcmBlock(tail, kWindowStride, sample_index);
    } else {
      ReadPcmBlock(i2s_raw, tail, kWindowStride);
    }
    sample_index += kWindowStride;

    // --- rebuild the complex-interleaved window ---------------------------
    for (size_t i = 0; i < win_len; ++i) {
      audio_win[i * 2] = pcm_win[i];
      audio_win[i * 2 + 1] = 0;
    }

    // --- one window -> one feature row ------------------------------------
    std::span<int8_t> row = {features + head * n_mel, n_mel};
    esp_err_t fe = extract_features(audio_win, scratch_buffer, row, fc,
                                    &profiler, 1);
    if (fe != ESP_OK) {
      MicroPrintf("extract_features failed: %d", (int)fe);
      return -1;
    }
    profiler.ClearEvents();

    int8_t row_min = INT8_MAX, row_max = INT8_MIN;
    for (size_t i = 0; i < n_mel; ++i) {
      row_min = std::min(row_min, row[i]);
      row_max = std::max(row_max, row[i]);
    }

    // In self-test mode, dump one settled row for comparison against Python's
    // process_window() on the same synthetic buffer, then keep looping.
    if (kSelfTest && !self_test_reported && filled + 1 >= n_windows) {
      MicroPrintf("Self-test feature row (%u int8 values):", (unsigned)n_mel);
      for (size_t i = 0; i < n_mel; ++i) {
        printf("%d%s", (int)row[i], (i + 1 < n_mel) ? "," : "\n");
      }
      self_test_reported = true;
    }

    head = (head + 1) % n_windows;
    if (filled < n_windows) ++filled;
    ++since_classify;

    // --- classify only on a primed buffer, every kClassifyHop windows ------
    if (filled < n_windows || since_classify < kClassifyHop) {
      continue;
    }
    since_classify = 0;

    // Copy rows into the input tensor in time order. `head` is the oldest row.
    int8_t* dst = tflite::GetTensorData<int8_t>(model_input);
    const size_t first_len = (n_windows - head) * n_mel;
    memcpy(dst, features + head * n_mel, first_len);
    memcpy(dst + first_len, features, head * n_mel);

    status = interpreter.Invoke();
    if ((status != kTfLiteOk) && (static_cast<int>(status) != kTfLiteAbort)) {
      MicroPrintf("Model interpreter invocation failed: %d", status);
      return -1;
    }
    profiler.ClearEvents();

    // --- report -----------------------------------------------------------
    const uint32_t rms = ComputeRms(tail, kWindowStride);
    const int8_t* out_data = tflite::GetTensorData<int8_t>(out_tensor);

    float p_target = 0.0f;
    printf("RMS=%6" PRIu32 "  FEAT[%4d..%4d]  p=[", rms, (int)row_min, (int)row_max);
    for (size_t i = 0; i < n_out; ++i) {
      const float p = (out_data[i] - out_zp) * out_scale;
      if (i == target_class) p_target = p;
      printf("%.3f%s", p, (i + 1 < n_out) ? " " : "");
    }
    printf("]");

    if (p_target >= kDetectThreshold) {
      ++consecutive_hits;
    } else {
      consecutive_hits = 0;
    }
    if (consecutive_hits >= kDetectConsecutive) {
      printf("   *** DETECTION (%d consecutive) ***", consecutive_hits);
    }
    printf("\n");
  }

  return -1;  // unreachable
}
}  // namespace
}  // namespace tflite


extern "C" void app_main() {
  // Give the USB-Serial/JTAG host time to enumerate before writing anything;
  // writes issued before enumeration are dropped by the lossy secondary sink
  // and never reach the terminal.
  vTaskDelay(pdMS_TO_TICKS(5000));

  MicroPrintf("\nConfigured arena size = %d\n", tflite::kTensorArenaSize);
  auto res = tflite::Benchmark(g_model, feature_config);
  MicroPrintf("Result=%d", res);

  // Only reached if Benchmark() hit an error and returned early (the success
  // path streams forever internally). Keep the console alive so the error
  // above can be read.
  while (true) {
    vTaskDelay(portMAX_DELAY);
  }
}