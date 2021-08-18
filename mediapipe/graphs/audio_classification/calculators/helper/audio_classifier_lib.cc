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

tflite::support::StatusOr<std::string> Classify_(
    const std::string& model_path, const mediapipe::Matrix& audio_data) {

    AudioClassifierOptions options;
    options.mutable_base_options()->mutable_model_file()->set_file_name(
        model_path);

    ASSIGN_OR_RETURN(std::unique_ptr<AudioClassifier> classifier,
                     AudioClassifier::CreateFromOptions(options));

    // `wav_data` holds data loaded from the file and needs to outlive `buffer`.
    std::vector<float> wav_data;
    wav_data.resize(audio_data.cols());
    Eigen::Map<Eigen::ArrayXf>(wav_data.data(), wav_data.size()) = audio_data.row(0);

    uint32 decoded_sample_count = wav_data.size();
    uint16 decoded_channel_count = 1;
    uint32 decoded_sample_rate = 16000;

    int buffer_size = classifier->GetRequiredInputBufferSize();

    if (decoded_sample_count > buffer_size) {
        decoded_sample_count = buffer_size;
    }

    AudioBuffer buffer = AudioBuffer(
        wav_data.data(), buffer_size,
        {decoded_channel_count, static_cast<int>(decoded_sample_rate)});

    ASSIGN_OR_RETURN(ClassificationResult result, classifier->Classify(buffer));
    const auto& head = result.classifications(0);
    const int score = head.classes(0).score();
    const std::string classification = head.classes(0).class_name();

    return classification;
}

}   // namespace audio
}   // namespace task
}   // namespace tflite
