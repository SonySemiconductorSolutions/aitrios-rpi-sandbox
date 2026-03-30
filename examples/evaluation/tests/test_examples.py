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

"""Simple tests for examples. Run with: uv run pytest tests/test_examples.py -v"""

from pathlib import Path

import pytest

from evaluator.common import load_custom_model_file

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = [
    "custom_cls_evaluation.py",
    "custom_od_evaluation.py",
]
SCRIPT_FILES = [
    "object_detection_aicamera_evaluator.py",
]


# -----------------------------------------------------------------------------
# Load and contract verification
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_loads_and_has_contract(filename):
    """Each example loads and defines the evaluator contract."""
    path = EXAMPLES_DIR / filename
    assert path.exists(), f"Example {filename} not found"
    mod = load_custom_model_file(path)
    assert hasattr(mod, "MODEL_PATH")
    assert hasattr(mod, "DATASET_PATH")
    assert hasattr(mod, "GROUND_TRUTH_PATH")
    assert hasattr(mod, "MODEL_CLASS")
    assert callable(getattr(mod, "get_dataset", None)) or callable(getattr(mod, "get_dataset_id_function", None)), (
        f"{filename} must define get_dataset or get_dataset_id_function"
    )


def test_examples_using_zoo_assets_have_examples_assets_paths():
    """Examples that use zoo assets (coco) reference examples/assets."""
    # custom_od_evaluation uses custom paths; custom_cls uses custom paths.
    # Check that default paths in common.py reference examples/assets.
    from evaluator.common import DEFAULT_DATASET, DEFAULT_GROUND_TRUTH

    for task_type, ds in DEFAULT_DATASET.items():
        assert "examples/assets" in ds, f"DEFAULT_DATASET[{task_type}]={ds!r} should contain examples/assets"
    for task_type, gt in DEFAULT_GROUND_TRUTH.items():
        assert "examples/assets" in gt, f"DEFAULT_GROUND_TRUTH[{task_type}]={gt!r} should contain examples/assets"


# -----------------------------------------------------------------------------
# custom_cls_evaluation: get_dataset_id_function
# -----------------------------------------------------------------------------


def test_example_custom_cls_get_dataset_id_function():
    """Classification id = parent dir name (class id)."""
    mod = load_custom_model_file(EXAMPLES_DIR / "custom_cls_evaluation.py")
    id_fn = mod.get_dataset_id_function("/tmp/gt")
    assert id_fn(Path("/tmp/gt/0/img.jpg")) == 0
    assert id_fn(Path("/tmp/gt/1/cat.png")) == 1


# -----------------------------------------------------------------------------
# custom_od_evaluation: _coco_image_id_from_filename
# -----------------------------------------------------------------------------


def test_example_custom_od_coco_image_id_from_filename():
    """COCO image_id lookup by filename."""
    mod = load_custom_model_file(EXAMPLES_DIR / "custom_od_evaluation.py")
    coco = {
        "images": [
            {"id": 42, "file_name": "a/b/foo.jpg"},
            {"id": 99, "file_name": "bar.png"},
        ]
    }
    assert mod._coco_image_id_from_filename("foo.jpg", coco) == 42
    assert mod._coco_image_id_from_filename("bar.png", coco) == 99


def test_example_custom_od_coco_image_id_not_found_raises():
    """Missing filename raises KeyError."""
    mod = load_custom_model_file(EXAMPLES_DIR / "custom_od_evaluation.py")
    coco = {"images": [{"id": 1, "file_name": "other.jpg"}]}
    with pytest.raises(KeyError, match="Image not found"):
        mod._coco_image_id_from_filename("missing.jpg", coco)


# -----------------------------------------------------------------------------
# Script examples (importable, no device)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("filename", SCRIPT_FILES)
def test_example_script_imports(filename):
    """Example scripts can be loaded as modules (no main block execution)."""
    import importlib.util

    path = EXAMPLES_DIR / filename
    assert path.exists(), f"Example {filename} not found"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
