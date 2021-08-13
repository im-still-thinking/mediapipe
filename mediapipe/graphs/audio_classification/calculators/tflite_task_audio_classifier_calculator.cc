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

#include <cstddef>
#include <iostream>
#include <limits>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/graphs/audio_classification/calculators/helper/audio_classifier_lib.h"

namespace mediapipe {

class TfliteTaskAudioClassifierCalculator : public CalculatorBase {
  public:
    TfliteTaskAudioClassifierCalculator() = default;

    static absl::Status GetContract(CalculatorContract* cc);
    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override {
        return mediapipe::tool::StatusStop();
    }
};

REGISTER_CALCULATOR(TfliteTaskAudioClassifierCalculator);

absl::Status TfliteTaskAudioClassifierCalculator::GetContract(CalculatorContract* cc) {
    RET_CHECK(!cc->InputSidePackets().GetTags().empty());
    cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
    cc->InputSidePackets().Tag("DATA_PATH").Set<std::string>();
    cc->OutputSidePackets().Tag("CLASS").Set<std::vector<std::string>>();

    return absl::OkStatus();
}

absl::Status TfliteTaskAudioClassifierCalculator::Open(CalculatorContext* cc) {
    const std::string& input_file_path =
        cc->InputSidePackets().Tag("DATA_PATH").Get<std::string>();
    const std::string& yamnet_model_path =
        cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();

    // const int min_score_thres = 0.5;

    //Start Classification
    tflite::support::StatusOr<std::vector<std::string>> result = tflite::task::audio::Classify(
        yamnet_model_path, input_file_path);
    if (result.ok()) {
        auto result_ = result.value();
        cc->OutputSidePackets().Tag("CLASS").Set(MakePacket<std::vector<std::string>>(result_));

    } else {
        std::cerr << "Classification failed: " << result.status().message() << "\n";
    }
    return absl::OkStatus();
}

}   // namespace mediapipe
