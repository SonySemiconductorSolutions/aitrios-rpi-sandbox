#
# Copyright 2026 Sony  Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
Custom model file contract for evaluator CLI (aicamera and interpreter).

Based on models trained with [aitrios-rpi-tutorials-ai-model-training](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training) (NanoDet object detector, e.g. PPE).

Define MODEL_PATH, DATASET_PATH, GROUND_TRUTH_PATH, MODEL_CLASS, and
get_dataset_id_function(ground_truth_path). Optional: get_dataset(dataset_path, ground_truth_path)
to build the Dataset yourself; MODEL_URI (or --model-file) for interpreter; get_evaluator(...) for custom class counts.

MODEL_URI (interpreter): path to the model in the container (or use --model-file to override). The interpreter runs in
a container; if the host models folder is mounted at /models in the container, use e.g. /models/ppe/nanodet_ppe.

Usage:
  uv run evaluator aicamera --model-config examples/custom_od_evaluation.py
  uv run evaluator interpreter --model-config examples/custom_od_evaluation.py
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from modlib.devices import Dataset
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.evals import COCOEvaluator
from modlib.models.zoo import NanoDetPlus416x416

# ---- Paths (defaults; CLI can override with --model-path, --dataset, --ground-truth) ----
MODEL_PATH = Path.home() / "models/nanodet-custom-ppe/packerOut.zip"
DATASET_PATH = Path.home() / "datasets/ppe/PPE-Detection-Using-CV-3/valid/"
GROUND_TRUTH_PATH = Path.home() / "datasets/ppe/PPE-Detection-Using-CV-3/valid/_annotations.coco.json"

# Optional: for interpreter — path inside the container (host models often mounted at /models)
DOCKER_MODEL_ROOT = "/models"
MODEL_URI = f"{DOCKER_MODEL_ROOT}/nanodet-custom-ppe/nanodet-quant-ppe.keras"  # e.g. "/models/ppe/nanodet_ppe" (container path, not host path)


def _coco_image_id_from_filename(image_filename: str, coco_data: dict[str, Any]) -> int:
    """
    Helper function to get the image_id from the filename.
    In this example its used to get image id:s from Roboflow coco format dataset file.
    Typical layout for Roboflow coco format dataset is:
    ./PPE-Detection-Using-CV-3
    ├─ README.dataset.txt
    ├─ README.roboflow.txt
    ./PPE-Detection-Using-CV-3/valid
    ├─ ppe_0872_jpg.rf.2b883f51d3f4da66fa040ba746923551.jpg
    ├─ image_14_jpg.rf.8e782445265350c3f3ad5b0481400da3.jpg
    ├─ ppe_1202_jpg.rf.ad7716edf2c9eedcdf41d4c3e2dc38ff.jpg
    ...
    ./PPE-Detection-Using-CV-3/train
    ├─ ppe_0477_jpg.rf.3d972c5b81169d745c2ba01d5fbe3620.jpg
    ├─ image_203_jpg.rf.bdf8d77ee06b5f8cd5efcce9ea39a731.jpg
    ├─ ppe_0874_jpg.rf.87ee89fd8d8bccbf8f1de62f942aa0d9.jpg
    ...

    Consequently, the image_id is NOT the same as the image filename,
    and therefore we need to use this helper function to get the
    image_id from the filename.
    """
    target_base = os.path.basename(image_filename)
    for img in coco_data["images"]:
        fn = img.get("file_name", "")
        if fn == image_filename or os.path.basename(fn) == target_base:
            return int(img["id"])
    raise KeyError(f"Image not found in COCO images[]: {image_filename}")


def get_dataset_id_function(ground_truth_path: str | Path):
    """Return a callable path -> image_id for use as Dataset dataset_id_function."""
    with open(ground_truth_path, encoding="utf-8") as f:
        coco = json.load(f)

    def dataset_id_function(path):
        return _coco_image_id_from_filename(str(path), coco)

    return dataset_id_function


def get_dataset(dataset_path: str | Path, ground_truth_path: str | Path) -> Dataset:
    """Build Dataset so each sample’s image_id comes from the COCO ground truth (same as this script).
    The runner receives this dataset and the evaluator uses those ids to match detections to ground truth.
    """
    id_fn = get_dataset_id_function(ground_truth_path)
    return Dataset(images_dir=Path(dataset_path), dataset_id_function=id_fn)


def get_evaluator(model: Model, ground_truth_path: str | Path):
    """Return evaluator for this custom detection model (8 classes, no COCO 80/91 mapping)."""
    return COCOEvaluator(ground_truth=ground_truth_path, label_mapping_func=(lambda x: x))


# ---- Model class ----
class NanoDetPPE(NanoDetPlus416x416):
    """Custom NanoDet with PPE classes. See aitrios-rpi-tutorials-ai-model-training."""

    def __init__(self, model_path: str | Path):
        Model.__init__(
            self,
            model_file=str(model_path),
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.BGR,
            preserve_aspect_ratio=False,
        )
        self.labels = np.array(
            [
                "safety-equipment",
                "person",
                "goggles",
                "helmet",
                "no-goggles",
                "no-helmet",
                "no-vest",
                "vest",
            ]
        )
        # Same as zoo NanoDetPlus416x416; needed for --prep model / --prep like_isp
        self.norm_mean = np.array([103.53, 116.28, 123.675])
        self.norm_std = np.array([57.375, 57.12, 58.395])


MODEL_CLASS = NanoDetPPE

if __name__ == "__main__":
    print("This module is a contract for evaluator CLI (aicamera and interpreter). See top of file for usage.")
    print("Basically: uv run evaluator <aicamera|interpreter> --model-config <this file> [--model-file <model path>]")
