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

"""Tests for checkpoint/resume (ID-based, strict config). Run with: uv run pytest tests/ -v"""

import pickle
import tempfile
from pathlib import Path

import pytest

from evaluator.common import (
    CHECKPOINT_VERSION,
    EVALUATION_PICKLE_KEYS,
    check_meta_compatible,
    filter_unprocessed,
    load_checkpoint,
    load_evaluation_pickle,
    make_checkpoint_meta,
    save_checkpoint,
)

# -----------------------------------------------------------------------------
# Helpers (no modlib)
# -----------------------------------------------------------------------------


def _ns(**kwargs):
    return type("Args", (), kwargs)()


class _FakeSample:
    """Picklable fake sample for tests."""

    def __init__(self, id_):
        self.id = id_


def _fake_meta(**overrides):
    base = {
        "version": CHECKPOINT_VERSION,
        "mode": "aicamera",
        "model_alias": "ssd",
        "model_class": "SSDMobileNetV2FPNLite320x320",
        "task_type": "TASK_TYPE.DETECTION",
        "dataset_path": str(Path("/data/images").resolve()),
        "ground_truth_path": str(Path("/data/gt.json").resolve()),
        "prep": "model",
        "extra_args": {"frame_rate": 5},
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# save / load round-trip
# -----------------------------------------------------------------------------


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "chk.pkl"
        meta = _fake_meta()
        processed_ids = {"a", "b", "c"}
        samples = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        save_checkpoint(path, meta, processed_ids, samples)
        loaded = load_checkpoint(path)
        assert loaded is not None
        assert loaded["meta"] == meta
        assert set(loaded["processed_ids"]) == processed_ids
        assert len(loaded["samples"]) == 3
        assert loaded["samples"][0]["id"] == "a"


def test_load_nonexistent_returns_none():
    with tempfile.TemporaryDirectory() as d:
        out = load_checkpoint(Path(d) / "nonexistent.pkl")
    assert out is None


def test_load_invalid_format_raises():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.pkl"
        path.write_bytes(b"not a valid checkpoint dict")
        with pytest.raises(ValueError, match="Invalid"):
            load_checkpoint(path)


# -----------------------------------------------------------------------------
# load_evaluation_pickle (standardized format for evaluators)
# -----------------------------------------------------------------------------


def test_load_evaluation_pickle_standardized():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "eval.pkl"
        meta = _fake_meta()
        save_checkpoint(path, meta, ["a", "b"], [{"id": "a"}, {"id": "b"}])
        out = load_evaluation_pickle(path)
    assert out is not None
    assert set(out.keys()) == set(EVALUATION_PICKLE_KEYS)
    assert out["meta"] == meta
    assert out["processed_ids"] == ["a", "b"]
    assert len(out["samples"]) == 2


def test_load_evaluation_pickle_legacy_list():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.pkl"
        samples = [{"id": 1}, {"id": 2}]
        with open(path, "wb") as f:
            pickle.dump(samples, f)
        out = load_evaluation_pickle(path)
    assert out is not None
    assert set(out.keys()) == set(EVALUATION_PICKLE_KEYS)
    assert out["meta"] is None
    assert out["processed_ids"] == []
    assert out["samples"] == samples


def test_load_evaluation_pickle_nonexistent_returns_none():
    with tempfile.TemporaryDirectory() as d:
        out = load_evaluation_pickle(Path(d) / "nonexistent.pkl")
    assert out is None


def test_load_evaluation_pickle_unsupported_format_raises():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.pkl"
        with path.open("wb") as f:
            pickle.dump({"only": "one_key"}, f)
        with pytest.raises(ValueError, match="Unsupported pickle format"):
            load_evaluation_pickle(path)


# -----------------------------------------------------------------------------
# make_checkpoint_meta
# -----------------------------------------------------------------------------


def test_make_checkpoint_meta_builds_consistent_meta():
    args = _ns(model="ssd", prep="model", frame_rate=5)
    model = type("M", (), {"task_type": "DETECTION", "__class__": type("M", (), {"__name__": "SSD"})})()
    meta = make_checkpoint_meta(
        mode="aicamera",
        dataset_path="/data/img",
        ground_truth_path="/data/gt.json",
        args=args,
        model=model,
    )
    assert meta["version"] == CHECKPOINT_VERSION
    assert meta["mode"] == "aicamera"
    assert meta["model_alias"] == "ssd"
    assert meta["prep"] == "model"
    assert "dataset_path" in meta and "ground_truth_path" in meta
    assert meta.get("extra_args", {}).get("frame_rate") == 5


# -----------------------------------------------------------------------------
# check_meta_compatible (strict)
# -----------------------------------------------------------------------------


def test_check_meta_compatible_same_ok():
    m = _fake_meta()
    check_meta_compatible(m, m)
    check_meta_compatible(m, _fake_meta())


def test_check_meta_compatible_different_model_alias_raises():
    a = _fake_meta(model_alias="ssd")
    b = _fake_meta(model_alias="yolo")
    with pytest.raises(ValueError, match="model_alias.*differs"):
        check_meta_compatible(a, b)


def test_check_meta_compatible_different_dataset_raises():
    a = _fake_meta(dataset_path="/data/a")
    b = _fake_meta(dataset_path="/data/b")
    with pytest.raises(ValueError, match="dataset_path.*differs"):
        check_meta_compatible(a, b)


def test_check_meta_compatible_different_prep_raises():
    a = _fake_meta(prep="model")
    b = _fake_meta(prep="isp")
    with pytest.raises(ValueError, match="prep.*differs"):
        check_meta_compatible(a, b)


def test_check_meta_compatible_different_frame_rate_ok():
    """frame_rate is ignored for resume; only evaluation-relevant config is checked."""
    a = _fake_meta(extra_args={"frame_rate": 5})
    b = _fake_meta(extra_args={"frame_rate": 10})
    check_meta_compatible(a, b)


# -----------------------------------------------------------------------------
# filter_unprocessed
# -----------------------------------------------------------------------------


def test_filter_unprocessed_yields_only_unprocessed():
    items = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
    processed = {1, 3, 5}
    out = list(filter_unprocessed(items, processed, lambda x: x["id"]))
    assert [x["id"] for x in out] == [2, 4]


def test_filter_unprocessed_empty_processed_yields_all():
    items = [{"id": 1}, {"id": 2}]
    out = list(filter_unprocessed(items, set(), lambda x: x["id"]))
    assert len(out) == 2


def test_filter_unprocessed_all_processed_yields_nothing():
    items = [{"id": 1}, {"id": 2}]
    out = list(filter_unprocessed(items, {1, 2}, lambda x: x["id"]))
    assert len(out) == 0


# -----------------------------------------------------------------------------
# Resume simulation (no real device/modlib)
# -----------------------------------------------------------------------------


def test_resume_simulation():
    """Simulate: run 2 samples → save checkpoint → resume → process remaining 3 → total 5."""

    def id_fn(s):
        return s.id

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "resume.pkl"

        # "Dataset": 5 samples with ids 0..4
        all_items = [_FakeSample(i) for i in range(5)]

        # First run: process 0, 1
        meta = _fake_meta(dataset_path=d, ground_truth_path=d)
        processed_ids = {0, 1}
        samples = [_FakeSample(0), _FakeSample(1)]
        save_checkpoint(path, meta, processed_ids, samples)

        # Resume: load + only iterate unprocessed
        cp = load_checkpoint(path)
        assert cp is not None
        check_meta_compatible(cp["meta"], meta)
        processed_ids = set(cp["processed_ids"])
        samples = list(cp["samples"])

        remaining = list(filter_unprocessed(all_items, processed_ids, id_fn))
        assert len(remaining) == 3
        assert [s.id for s in remaining] == [2, 3, 4]

        # Append "new" results for 2, 3, 4
        for s in remaining:
            samples.append(s)
            processed_ids.add(id_fn(s))

        assert len(samples) == 5
        assert processed_ids == {0, 1, 2, 3, 4}
