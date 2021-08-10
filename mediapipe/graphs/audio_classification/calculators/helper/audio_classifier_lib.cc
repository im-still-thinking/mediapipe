/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/graphs/audio_classification/calculators/helper/audio_classifier_lib.h"
#include "mediapipe/graphs/audio_classification/calculators/helper/wav/wav_io.h"

#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/audio/audio_classifier.h"
#include "tensorflow_lite_support/cc/task/audio/core/audio_buffer.h"
#include "tensorflow_lite_support/cc/task/audio/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/core/category.h"

namespace tflite {
namespace task {
namespace audio {

tflite::support::StatusOr<ClassificationResult> Classify(
    const std::string& model_path, const std::string& wav_file,
    bool use_coral) {
    AudioClassifierOptions options;
    options.mutable_base_options()->mutable_model_file()->set_file_name(
        model_path);
    if (use_coral) {
        options.mutable_base_options()
            ->mutable_compute_settings()
            ->mutable_tflite_settings()
            ->set_delegate(::tflite::proto::Delegate::EDGETPU_CORAL);
    }
    ASSIGN_OR_RETURN(std::unique_ptr<AudioClassifier> classifier,
                     AudioClassifier::CreateFromOptions(options));

    // `wav_data` holds data loaded from the file and needs to outlive `buffer`.
    std::vector<float> wav_data;

    //Load data from wav file
    std::string contents = ReadFile(wav_file);
    uint32 decoded_sample_count;
    uint16 decoded_channel_count;
    uint32 decoded_sample_rate;

    int buffer_size = classifier->GetRequiredInputBufferSize();
    RETURN_IF_ERROR(DecodeLin16WaveAsFloatVector(
        contents, &wav_data, &decoded_sample_count, &decoded_channel_count,
        &decoded_sample_rate));

    std::cout << "decoded_sample_count: " << decoded_sample_count << std::endl;

    std::cout << "buffer_size: " << buffer_size << std::endl;
    std::vector<AudioBuffer> buffer_array;

    std::vector<float>::iterator it = wav_data.begin();

    while (it != wav_data.end()) {
        std::vector<float> sliced_wav_data;

        if (it + buffer_size > wav_data.end()) {
            sliced_wav_data.assign(it, wav_data.end());
            it = wav_data.end();
        } else {
            sliced_wav_data.assign(it, it + buffer_size);
            it += buffer_size;
        }

        AudioBuffer buffer = AudioBuffer(
            sliced_wav_data.data(), buffer_size,
            {decoded_channel_count, static_cast<int>(decoded_sample_rate)});

        buffer_array.push_back(buffer);
    }

    auto start_classify = std::chrono::steady_clock::now();
    std::vector<ClassificationResult> results;
    for (auto buff : buffer_array) {
        ASSIGN_OR_RETURN(ClassificationResult result, classifier->Classify(buff));
        results.push_back(result);
    }

    auto end_classify = std::chrono::steady_clock::now();
    std::string delegate = use_coral ? "Coral Edge TPU" : "CPU";
    const auto duration_ms =
        std::chrono::duration<float, std::milli>(end_classify - start_classify);
    std::cout << "Time cost to classify the input audio clip on " << delegate
              << ": " << duration_ms.count() << " ms" << std::endl;

    return results[0];
}

}   // namespace audio
}   // namespace task
}   // namespace tflite
