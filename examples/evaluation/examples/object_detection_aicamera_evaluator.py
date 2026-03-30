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
Simple example showing how to use the AiCamera evaluator programmatically.

This example demonstrates:
- Loading a model from the model zoo
- Setting up dataset and evaluator with custom paths
- Running data injection evaluation
- Getting evaluation results

Modify the paths below to match your dataset locations.

Run with:
    uv run examples/object_detection_aicamera_evaluator.py
"""

from modlib.devices import AiCamera

from evaluator.aicamera import MODELS, run_data_injection
from evaluator.common import get_dataset, get_evaluator

if __name__ == "__main__":
    # Configuration - modify these paths to match your setup
    DATASET_PATH = "./examples/assets/coco_samples/"  # Set to your dataset path
    GROUND_TRUTH_PATH = "./examples/assets/coco_annotations/instances_val2017.json"  # Set to your ground truth path
    MODEL_NAME = "ssd"  # Options: "ssd", "yolo", "nano", "effdet", etc, see aliases in evaluator.aicamera

    # 1. Initialize model
    model = MODELS[MODEL_NAME]()  # Use SSD MobileNet model

    # 2. Get dataset and evaluator with custom paths
    # If paths are None, will use defaults (may not exist)
    dataset = get_dataset(model, dataset=DATASET_PATH)
    evaluator = get_evaluator(model, ground_truth=GROUND_TRUTH_PATH)

    # 3. Initialize device
    device = AiCamera(data_injection=True, frame_rate=5, enable_input_tensor=False)
    device.deploy(model)

    # 4. Run data injection evaluation
    # Create a simple args object for preprocessing method
    class Args:
        prep = "model"  # Options: "model", "like_isp", "isp"
        display_image = False
        display_input_tensor = False

    args = Args()
    all_samples = run_data_injection(
        model=model,
        device=device,
        dataset=dataset,
        args=args,
    )

    # 5. Evaluate results
    evaluator.evaluate(all_samples)
    evaluator.visualize(all_samples, output_dir="tmp/annotated_images")

    print("Evaluation complete!")
