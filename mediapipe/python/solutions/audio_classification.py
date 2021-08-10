# Copyright 2021 The MediaPipe Authors.
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

"""MediaPipe Audio Classification."""

import enum
from typing import NamedTuple, Union

import numpy as np
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import download_utils

BINARYPB_FILE_PATH = 'mediapipe/modules/audio_classification/audio_classification_desktop_live.binarypb'
YAMNET_MODEL_PATH = 'mediapipe/modules/audio_classification/audio_classification_desktop_live.binarypb'


def _download_oss_pose_landmark_model():
  """Downloads the pose landmark lite/heavy model from the MediaPipe Github repo if it doesn't exist in the package."""
  download_utils.download_oss_model(
      'mediapipe/modules/audio_classification/yamnet.tflite')


class AudioEventClassifier(SolutionBase):

  def __init__(self, input_wav_file: str):

    _download_oss_pose_landmark_model()
    super().__init__(binary_graph_path=BINARYPB_FILE_PATH,
                     side_inputs={
                         'yamnet_model_path': YAMNET_MODEL_PATH,
                         'input_audio_wav_path': input_wav_file
                     },
                     outputs=['audio_class'])

  def process(self) -> str:
    return super().process()
