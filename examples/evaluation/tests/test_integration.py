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
Minimum integration tests. Run with: uv run pytest tests/test_integration.py -v

These tests exercise real code paths with assets and/or network. Some are skipped
when required resources (e.g. examples/assets) are not present.
"""

import pickle
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

# Paths relative to tools/evaluation
ASSETS_DIR = Path(__file__).resolve().parent.parent / "examples" / "assets"
COCO_SAMPLES = ASSETS_DIR / "coco_samples"
COCO_GT = ASSETS_DIR / "coco_annotations" / "instances_val2017.json"


def _assets_exist() -> bool:
    return COCO_SAMPLES.is_dir() and COCO_GT.is_file()


@pytest.fixture
def eval_root():
    """Root directory for evaluator (tools/evaluation)."""
    return Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Integration: build model, dataset, evaluator (no device)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _assets_exist(), reason="examples/assets not present (run fetch_modlib_assets.sh)")
def test_integration_build_model_dataset_evaluator(eval_root):
    """Build model, dataset, and evaluator using real assets."""
    import sys

    sys.path.insert(0, str(eval_root / "src"))
    from evaluator.aicamera import MODELS
    from evaluator.common import get_dataset, get_evaluator

    model = MODELS["ssd"]()
    dataset = get_dataset(model, str(COCO_SAMPLES))
    evaluator = get_evaluator(model, str(COCO_GT))

    assert len(list(dataset)) > 0
    assert evaluator is not None
    assert str(evaluator.ground_truth) == str(COCO_GT.resolve())


# -----------------------------------------------------------------------------
# Integration: check_interpreter_reachable with real HTTP server
# -----------------------------------------------------------------------------


def test_integration_check_interpreter_reachable_success(eval_root):
    """check_interpreter_reachable succeeds when endpoint responds."""
    import sys

    sys.path.insert(0, str(eval_root / "src"))

    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        from evaluator.interpreter import check_interpreter_reachable

        # Should not sys.exit when endpoint is reachable
        check_interpreter_reachable(f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()


def test_integration_check_interpreter_reachable_fails_unreachable(eval_root):
    """check_interpreter_reachable exits when endpoint is unreachable."""
    import sys

    sys.path.insert(0, str(eval_root / "src"))

    with pytest.raises(SystemExit) as exc_info:
        from evaluator.interpreter import check_interpreter_reachable

        check_interpreter_reachable("http://127.0.0.1:19999", timeout=0.5)
    assert exc_info.value.code == 1


# -----------------------------------------------------------------------------
# Integration: evaluator --load-samples (full path without device)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _assets_exist(), reason="examples/assets not present (run fetch_modlib_assets.sh)")
def test_integration_load_samples_evaluate(eval_root):
    """Run evaluator aicamera --load-samples with a pre-built pickle (no device)."""
    from modlib.devices import Dataset
    from modlib.models import ROI
    from modlib.models.evals import EvaluationSample
    from modlib.models.results import Detections

    # Build one EvaluationSample from real dataset (need ≥1 detection for COCO evaluate)
    dataset = Dataset(images_dir=str(COCO_SAMPLES))
    sample = next(iter(dataset))
    roi = ROI(left=0, top=0, width=1, height=1)
    # Minimal detection: one bbox [x, y, w, h] so pycocotools loadRes gets non-empty anns
    det = Detections(
        bbox=np.array([[10.0, 10.0, 50.0, 50.0]]),
        confidence=np.array([0.5]),
        class_id=np.array([0]),
    )
    ev_sample = EvaluationSample(roi=roi, detections=det, dataset_sample=sample)

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "samples.pkl"
        payload = {
            "meta": None,
            "processed_ids": [sample.image_id],
            "samples": [ev_sample],
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

        out_dir = Path(tmp) / "annotated"
        result = subprocess.run(
            [
                "uv",
                "run",
                "evaluator",
                "aicamera",
                "-m",
                "ssd",
                "--load-samples",
                "-p",
                str(pkl_path),
                "--dataset",
                str(COCO_SAMPLES),
                "--ground-truth",
                str(COCO_GT),
                "--annotated-output-dir",
                str(out_dir),
            ],
            cwd=str(eval_root),
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
