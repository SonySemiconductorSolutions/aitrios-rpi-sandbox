#
# Copyright 2026 Sony  Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Custom classification model file contract for evaluator CLI (aicamera and interpreter).

Based on models trained with [aitrios-rpi-tutorials-ai-model-training](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training) (MobileNetV2 classifier, e.g. Rock/Paper/Scissors).

Uses a MobileNetV2-based model. Dataset layout: <DATASET_PATH>/<class_id>/<images>
(e.g. 0/cat.jpg, 1/dog.jpg). Ground truth is the same directory; image_id = class_id.

Define MODEL_PATH, DATASET_PATH, GROUND_TRUTH_PATH, MODEL_CLASS, and
get_dataset_id_function(ground_truth_path). Optional: get_dataset(dataset_path, ground_truth_path);
MODEL_URI (or --model-file) for interpreter.

Usage:
  uv run evaluator aicamera --model-config examples/custom_cls_evaluation.py
  uv run evaluator interpreter --model-config examples/custom_cls_evaluation.py
"""

from pathlib import Path

import numpy as np
from modlib.devices import Dataset
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.evals import ClassificationEvaluator
from modlib.models.zoo import MobileNetV2

# ---- Paths (defaults; CLI can override with --model-file, --dataset, --ground-truth) ----
MODEL_PATH = Path.home() / "models/mobilenet-custom-rps/packerOut.zip"
# Classification: dataset and ground truth are the same dir, layout <class_id>/<images>
DATASET_PATH = Path.home() / "datasets/rock_paper_scissors/test"
GROUND_TRUTH_PATH = DATASET_PATH

# Optional: for interpreter — path inside the container
DOCKER_MODEL_ROOT = "/models"
MODEL_URI = f"{DOCKER_MODEL_ROOT}/mobilenet-custom-rps/mobilenet-quant-rps.keras"  # e.g. "/models/mobilenet_v2/mobilenet_v2.keras"


def get_dataset_id_function(ground_truth_path: str | Path):
    """Return path -> image_id. For classification, image_id = class_id = parent dir name."""

    def dataset_id_function(path):
        return int(Path(path).parent.name)

    return dataset_id_function


def get_dataset(dataset_path: str | Path, ground_truth_path: str | Path) -> Dataset:
    """Build Dataset; each sample's image_id is the class id from the directory structure."""
    id_fn = get_dataset_id_function(ground_truth_path)
    return Dataset(images_dir=Path(dataset_path), dataset_id_function=id_fn)


def get_evaluator(model: Model, ground_truth_path: str | Path):
    """Return evaluator for this custom model (3 classes)"""
    return ClassificationEvaluator(ground_truth=ground_truth_path, labels=model.labels)


# ---- Model class ----
class CustomMobileNetV2(MobileNetV2):
    """Custom MobileNetV2 classifier loaded from model_path. Uses ImageNet labels from zoo assets if present."""

    def __init__(self, model_path: str | Path):
        Model.__init__(
            self,
            model_file=str(model_path),
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        # Same norm as zoo MobileNetV2
        self.norm_mean = 127.5
        self.norm_std = 127.5
        # Custom labels for RPS
        self.labels = np.array(["Rock", "Paper", "Scissors"])


MODEL_CLASS = CustomMobileNetV2

if __name__ == "__main__":
    print("This module is a contract for evaluator CLI (aicamera and interpreter). See top of file for usage.")
    print("Basically: uv run evaluator <aicamera|interpreter> --model-config <this file> [--model-file <model path>]")
