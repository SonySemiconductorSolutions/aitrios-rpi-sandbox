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

"""Tests for common utilities. Run with: uv run pytest tests/ -v"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from modlib.models.results import Classifications, Detections

from evaluator.common import (
    DEFAULT_DATASET,
    DEFAULT_GROUND_TRUTH,
    MODEL_CONTAINER_PREFIX,
    MODEL_HOST_PREFIX,
    format_debug_results,
    host_path_to_container_uri,
    load_custom_model_file,
)

# -----------------------------------------------------------------------------
# Default asset paths (examples/assets)
# -----------------------------------------------------------------------------

ASSETS_PREFIX = "examples/assets"


def test_default_dataset_paths_include_examples_assets():
    """All DEFAULT_DATASET paths include examples/assets."""
    for task, path in DEFAULT_DATASET.items():
        assert ASSETS_PREFIX in path, f"DEFAULT_DATASET[{task}]={path!r} should contain {ASSETS_PREFIX!r}"


def test_default_ground_truth_paths_include_examples_assets():
    """All DEFAULT_GROUND_TRUTH paths include examples/assets."""
    for task, path in DEFAULT_GROUND_TRUTH.items():
        assert ASSETS_PREFIX in path, f"DEFAULT_GROUND_TRUTH[{task}]={path!r} should contain {ASSETS_PREFIX!r}"


def test_default_paths_match_expected_asset_subdirs():
    """Default paths reference expected asset subdirs (coco, imagenet, voc, keypoints)."""
    all_paths = list(DEFAULT_DATASET.values()) + list(DEFAULT_GROUND_TRUTH.values())
    expected_subdirs = ("coco_samples", "coco_annotations", "imagenet", "voc_samples", "coco_keypoints")
    found = [s for s in expected_subdirs if any(s in p for p in all_paths)]
    assert len(found) >= 2, f"Expected at least coco_samples and coco_annotations; found {found}"


# -----------------------------------------------------------------------------
# host_path_to_container_uri (interpreter model path mapping)
# -----------------------------------------------------------------------------


def test_host_path_to_container_uri_under_prefix():
    """Path under MODEL_HOST_PREFIX is translated to MODEL_CONTAINER_PREFIX."""
    prefix = Path(MODEL_HOST_PREFIX).resolve()
    host_path = prefix / "my_model" / "model.onnx"
    got = host_path_to_container_uri(str(host_path))
    assert got == f"{MODEL_CONTAINER_PREFIX}/my_model/model.onnx"


def test_host_path_to_container_uri_outside_prefix_returns_none():
    """Path not under MODEL_HOST_PREFIX returns None (caller uses path as container path)."""
    got = host_path_to_container_uri("/other/models/foo.onnx")
    assert got is None


def test_host_path_to_container_uri_relative_returns_none():
    """Relative path returns None."""
    assert host_path_to_container_uri("models/foo.onnx") is None


def test_host_path_to_container_uri_empty_returns_none():
    """Empty path returns None."""
    assert host_path_to_container_uri("") is None


def test_host_path_to_container_uri_container_path_returns_none():
    """Container path /models/... is not under host prefix, returns None (use as-is)."""
    assert host_path_to_container_uri("/models/foo/model.onnx") is None


def test_host_path_to_container_uri_custom_mapping():
    """Custom host_prefix and container_prefix override defaults."""
    # Under /opt/my_models -> /container/models
    got = host_path_to_container_uri(
        "/opt/my_models/foo/model.onnx",
        host_prefix="/opt/my_models",
        container_prefix="/container/models",
    )
    assert got == "/container/models/foo/model.onnx"


EVAL_ROOT = Path(__file__).resolve().parent.parent


def _default_paths_exist() -> bool:
    """True if assets from fetch_modlib_assets.sh are present."""
    # Use coco_samples as canary; if present, assume full fetch ran
    return (EVAL_ROOT / "examples" / "assets" / "coco_samples").is_dir()


@pytest.mark.skipif(
    not _default_paths_exist(),
    reason="examples/assets not present (run fetch_modlib_assets.sh)",
)
def test_default_paths_exist_and_match_fetch_assets():
    """DEFAULT_DATASET and DEFAULT_GROUND_TRUTH paths exist and match fetch_modlib_assets.sh layout.

    Ensures common.py stays in sync with what fetch_modlib_assets.sh produces.
    """
    for task, path in DEFAULT_DATASET.items():
        resolved = (EVAL_ROOT / path).resolve()
        assert resolved.exists(), (
            f"DEFAULT_DATASET[{task}]={path!r} -> {resolved} does not exist. "
            "Run fetch_modlib_assets.sh or ensure paths match modlib get_assets output."
        )
        assert resolved.is_dir(), f"DEFAULT_DATASET[{task}]={path!r} should be a directory"

    for task, path in DEFAULT_GROUND_TRUTH.items():
        resolved = (EVAL_ROOT / path).resolve()
        assert resolved.exists(), (
            f"DEFAULT_GROUND_TRUTH[{task}]={path!r} -> {resolved} does not exist. "
            "Run fetch_modlib_assets.sh or ensure paths match modlib get_assets output."
        )
        # DETECTION/POSE: json file; CLASSIFICATION/SEGMENTATION: directory
        if path.endswith(".json"):
            assert resolved.is_file(), f"DEFAULT_GROUND_TRUTH[{task}]={path!r} should be a file"
        else:
            assert resolved.is_dir(), f"DEFAULT_GROUND_TRUTH[{task}]={path!r} should be a directory"


# -----------------------------------------------------------------------------
# format_debug_results (uses modlib result types)
# -----------------------------------------------------------------------------


def test_format_debug_results_classifications_empty():
    r = Classifications(confidence=np.array([]), class_id=np.array([]))
    out = format_debug_results(r)
    assert out == "Classifications: 0"


def test_format_debug_results_classifications_simple():
    r = Classifications(
        confidence=np.array([0.9, 0.7]),
        class_id=np.array([0, 1]),
    )
    out = format_debug_results(r, labels=["cat", "dog"])
    assert "Classifications: 2" in out
    assert "cat" in out and "dog" in out


def test_format_debug_results_detections_empty():
    r = Detections(
        bbox=np.empty((0, 4)),
        confidence=np.array([]),
        class_id=np.array([]),
    )
    out = format_debug_results(r)
    assert out == "Detections: 0"


def test_format_debug_results_detections_simple():
    r = Detections(
        bbox=np.array([[10, 20, 50, 60]]),
        confidence=np.array([0.85]),
        class_id=np.array([0]),
    )
    out = format_debug_results(r, labels=["person"])
    assert "Detections: 1" in out
    assert "person" in out


# -----------------------------------------------------------------------------
# load_custom_model_file
# -----------------------------------------------------------------------------


def test_load_custom_model_file_valid():
    """Load a minimal valid model file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
MODEL_PATH = "/tmp/model.zip"
DATASET_PATH = "/tmp/dataset"
GROUND_TRUTH_PATH = "/tmp/gt.json"
MODEL_CLASS = None  # not used for load test

def get_dataset_id_function(gt_path):
    return lambda p: 0
"""
        )
        path = Path(f.name)
    try:
        mod = load_custom_model_file(path)
        assert mod.MODEL_PATH == "/tmp/model.zip"
        assert mod.DATASET_PATH == "/tmp/dataset"
        assert hasattr(mod, "get_dataset_id_function")
    finally:
        path.unlink()


def test_load_custom_model_file_nonexistent_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_custom_model_file("/nonexistent/path/model.py")
