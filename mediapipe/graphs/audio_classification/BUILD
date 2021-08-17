# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_classification_desktop_live",
    deps = [
        "//mediapipe/calculators/audio:audio_decoder_calculator",
        "//mediapipe/calculators/core:add_header_calculator",
        "//mediapipe/calculators/audio:basic_time_series_calculators",
        "//mediapipe/calculators/audio:rational_factor_resample_calculator",
        "//mediapipe/graphs/audio_classification/calculators:tflite_task_audio_classifier_calculator",
    ],
)

mediapipe_binary_graph(
    name = "audio_classification_desktop_live_binary_graph",
    graph = "audio_classification_desktop_live.pbtxt",
    output_name = "audio_classification_desktop_live.binarypb",
    deps = [":desktop_calculators"],
)
