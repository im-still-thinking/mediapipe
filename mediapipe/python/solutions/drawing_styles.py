# Copyright 2020 The MediaPipe Authors.
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

"""MediaPipe solution drawing styles."""

from typing import Mapping, Tuple
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

RED = (54, 67, 244)
GREEN = (118, 230, 0)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
GRAY = (174, 164, 144)
PURPLE = (251, 64, 224)
PEACH = (180, 229, 255)


def get_hand_landmark_connections_annotations() -> Tuple[Mapping[Tuple[int, int], DrawingSpec],
                                                         Mapping[int, DrawingSpec]]:
    """Provides styling data for annotations for hands detected on an RGB image.

    Returns:
      A Tuple object with a "connection_annotations" and "landmark_annotations" field 
      that contains the styling data for hands detected.
    """

    THICKNESS_WRIST_MCP = 3
    THICKNESS_FINGER = 2
    THICKNESS_DOT = -1
    RADIUS = 5

    connection_annotations = {
        (HandLandmark.WRIST, HandLandmark.THUMB_CMC): DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
        (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP): DrawingSpec(color=PEACH, thickness=THICKNESS_FINGER),
        (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP): DrawingSpec(color=PEACH, thickness=THICKNESS_FINGER),
        (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP): DrawingSpec(color=PEACH, thickness=THICKNESS_FINGER),
        (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP): DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
        (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP): DrawingSpec(color=PURPLE, thickness=THICKNESS_FINGER),
        (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP): DrawingSpec(color=PURPLE, thickness=THICKNESS_FINGER),
        (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP): DrawingSpec(color=PURPLE, thickness=THICKNESS_FINGER),
        (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP): DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
        (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP): DrawingSpec(color=YELLOW, thickness=THICKNESS_FINGER),
        (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP): DrawingSpec(color=YELLOW, thickness=THICKNESS_FINGER),
        (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP): DrawingSpec(color=YELLOW, thickness=THICKNESS_FINGER),
        (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP): DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
        (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP): DrawingSpec(color=GREEN, thickness=THICKNESS_FINGER),
        (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP): DrawingSpec(color=GREEN, thickness=THICKNESS_FINGER),
        (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP): DrawingSpec(color=GREEN, thickness=THICKNESS_FINGER),
        (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP): DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
        (HandLandmark.WRIST, HandLandmark.PINKY_MCP): DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
        (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP): DrawingSpec(color=BLUE, thickness=THICKNESS_FINGER),
        (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP): DrawingSpec(color=BLUE, thickness=THICKNESS_FINGER),
        (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP): DrawingSpec(color=BLUE, thickness=THICKNESS_FINGER)
    }

    landmark_annotations = {
        HandLandmark.WRIST: DrawingSpec(color=RED, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.THUMB_CMC: DrawingSpec(color=RED, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.THUMB_MCP: DrawingSpec(color=PEACH, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.THUMB_IP: DrawingSpec(color=PEACH, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.THUMB_TIP: DrawingSpec(color=PEACH, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.INDEX_FINGER_MCP: DrawingSpec(color=RED, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.INDEX_FINGER_PIP: DrawingSpec(color=PURPLE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.INDEX_FINGER_DIP: DrawingSpec(color=PURPLE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.INDEX_FINGER_TIP: DrawingSpec(color=PURPLE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.MIDDLE_FINGER_MCP: DrawingSpec(color=RED, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.MIDDLE_FINGER_PIP: DrawingSpec(color=YELLOW, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.MIDDLE_FINGER_DIP: DrawingSpec(color=YELLOW, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.MIDDLE_FINGER_TIP: DrawingSpec(color=YELLOW, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.RING_FINGER_MCP: DrawingSpec(color=RED, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.RING_FINGER_PIP: DrawingSpec(color=GREEN, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.RING_FINGER_DIP: DrawingSpec(color=GREEN, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.RING_FINGER_TIP: DrawingSpec(color=GREEN, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.PINKY_MCP: DrawingSpec(color=RED, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.PINKY_PIP: DrawingSpec(color=BLUE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.PINKY_DIP: DrawingSpec(color=BLUE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
        HandLandmark.PINKY_TIP: DrawingSpec(color=BLUE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
    }

    return (connection_annotations, landmark_annotations)
