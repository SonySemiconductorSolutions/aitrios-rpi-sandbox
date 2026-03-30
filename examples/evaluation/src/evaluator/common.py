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

import importlib.util
import os
import pickle
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from modlib.apps import Annotator
from modlib.devices import IMAGE_TYPE, Dataset, Frame
from modlib.models import COLOR_FORMAT, ROI, TASK_TYPE, Model
from modlib.models.evals import (
    ClassificationEvaluator,
    COCOEvaluator,
    COCOPoseEvaluator,
    Evaluator,
    VocSegEvaluator,
)
from modlib.models.results import Classifications, Detections, Poses, Segments

DEFAULT_DATASET = {
    TASK_TYPE.CLASSIFICATION: "./examples/assets/imagenet_dataset",
    TASK_TYPE.DETECTION: "./examples/assets/coco_samples",
    TASK_TYPE.POSE: "./examples/assets/coco_keypoints_samples",
    TASK_TYPE.SEGMENTATION: "./examples/assets/voc_samples/JPEGImages",
}

DEFAULT_GROUND_TRUTH = {
    TASK_TYPE.CLASSIFICATION: "./examples/assets/imagenet_dataset",
    TASK_TYPE.DETECTION: "./examples/assets/coco_annotations/instances_val2017.json",
    TASK_TYPE.POSE: "./examples/assets/coco_annotations/person_keypoints_val2017.json",
    TASK_TYPE.SEGMENTATION: "./examples/assets/voc_samples/SegmentationClass",
}

# Host path → container path mapping for interpreter model loading.
# Paths under MODEL_HOST_PREFIX on the host are translated to MODEL_CONTAINER_PREFIX in the container
# (e.g. mount ~/models at /models so the interpreter can load the file).
MODEL_HOST_PREFIX = os.path.expanduser("~/models")
MODEL_CONTAINER_PREFIX = "/models"


def host_path_to_container_uri(
    host_path: str,
    *,
    host_prefix: str | None = None,
    container_prefix: str | None = None,
) -> str | None:
    """If host_path is under host_prefix, return the corresponding container URI under container_prefix; else None.
    Caller should use the return value when non-None; otherwise treat the path as already a container path.
    Defaults: MODEL_HOST_PREFIX, MODEL_CONTAINER_PREFIX when host_prefix/container_prefix are None.
    """
    if not host_path or not os.path.isabs(host_path):
        return None
    path = Path(host_path).resolve()
    prefix = Path(os.path.expanduser(host_prefix or MODEL_HOST_PREFIX)).resolve()
    container = (container_prefix if container_prefix is not None else MODEL_CONTAINER_PREFIX).rstrip("/")
    try:
        rel = path.relative_to(prefix)
    except ValueError:
        return None
    return f"{container}/{rel.as_posix()}"


def get_dataset(model: Model, dataset: str | None = None) -> Dataset:
    """Get dataset for evaluation based on model task type."""
    if dataset is None:
        dataset = DEFAULT_DATASET[model.task_type]
    images_dir = Path(dataset)
    if not images_dir.exists():
        raise FileNotFoundError(f"\nThe path {images_dir} does not exist.\n")

    def id_fn(path):
        return int(path.parent.name)

    id_fn = id_fn if model.task_type == TASK_TYPE.CLASSIFICATION else None
    return Dataset(images_dir=images_dir, dataset_id_function=id_fn)


def get_evaluator(model: Model, ground_truth: str | None = None) -> Evaluator:
    """Get evaluator for evaluation based on model task type."""
    if ground_truth is None:
        ground_truth = DEFAULT_GROUND_TRUTH[model.task_type]
    gt_dir = Path(ground_truth)
    if not gt_dir.exists():
        raise FileNotFoundError(f"\nThe path {gt_dir} does not exist.\n")

    if model.task_type == TASK_TYPE.CLASSIFICATION:
        return ClassificationEvaluator(ground_truth=gt_dir, labels=model.labels)
    elif model.task_type == TASK_TYPE.DETECTION:
        return COCOEvaluator(ground_truth=gt_dir, num_classes=len(model.labels))
    elif model.task_type == TASK_TYPE.POSE:
        return COCOPoseEvaluator(ground_truth=gt_dir)
    elif model.task_type == TASK_TYPE.SEGMENTATION:
        return VocSegEvaluator(ground_truth=gt_dir, num_classes=len(model.labels))
    else:
        raise ValueError(f"Unsupported task type: {model.task_type}")


# -----------------------------------------------------------------------------
# Custom model file contract (one Python file with paths, model class, id function)
# -----------------------------------------------------------------------------


def load_custom_model_file(file_path: str | Path):
    """Load a Python file as a module. Expected names: MODEL_PATH, DATASET_PATH,
    GROUND_TRUTH_PATH, MODEL_CLASS, get_dataset_id_function(ground_truth_path).
    For interpreter, optional MODEL_URI.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load model file as module: {path}")
    module = importlib.util.module_from_spec(spec)
    if path.parent not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def build_model_dataset_evaluator_from_file(
    file_path: str | Path,
    *,
    model_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
):
    """Build (model, dataset, evaluator) from a custom model file.
    File must define MODEL_PATH, DATASET_PATH, GROUND_TRUTH_PATH, MODEL_CLASS,
    and get_dataset_id_function(ground_truth_path) -> callable (path -> image_id for COCO).
    Optional get_dataset(dataset_path, ground_truth_path) -> Dataset overrides dataset build
    (so the custom file owns how sample image_ids are set, as in custom_od_evaluation).
    Optional get_evaluator(model, ground_truth_path) overrides shared evaluator (e.g. for custom class counts).
    Pass model_path/dataset_path/ground_truth_path to override file defaults.
    """
    module = load_custom_model_file(file_path)
    mp = model_path if model_path is not None else getattr(module, "MODEL_PATH", None)
    dp = dataset_path if dataset_path is not None else getattr(module, "DATASET_PATH", None)
    gp = ground_truth_path if ground_truth_path is not None else getattr(module, "GROUND_TRUTH_PATH", None)
    if mp is None or dp is None or gp is None:
        raise ValueError(
            "Custom model file must define MODEL_PATH, DATASET_PATH, GROUND_TRUTH_PATH "
            "(or pass --model-file, --dataset, --ground-truth)"
        )
    model_class = getattr(module, "MODEL_CLASS", None)
    if model_class is None:
        raise ValueError("Custom model file must define MODEL_CLASS (the model class to instantiate)")
    model = model_class(mp)
    get_ds_fn = getattr(module, "get_dataset", None)
    if get_ds_fn is not None and callable(get_ds_fn):
        dataset = get_ds_fn(str(dp), str(gp))
    else:
        id_fn = getattr(module, "get_dataset_id_function", None)
        if id_fn is None or not callable(id_fn):
            raise ValueError(
                "Custom model file must define get_dataset_id_function(ground_truth_path) or get_dataset(dataset_path, ground_truth_path)"
            )
        dataset_id_fn = id_fn(gp)
        dataset = Dataset(images_dir=Path(dp), dataset_id_function=dataset_id_fn)
    get_eval_fn = getattr(module, "get_evaluator", None)
    if get_eval_fn is not None and callable(get_eval_fn):
        evaluator = get_eval_fn(model, str(gp))
    else:
        evaluator = get_evaluator(model, str(gp))
    return (model, dataset, evaluator)


def get_custom_model_uri(file_path: str | Path) -> str | None:
    """Return MODEL_URI from a custom model file, if defined. Used by interpreter."""
    module = load_custom_model_file(file_path)
    return getattr(module, "MODEL_URI", None)


def format_debug_results(
    results: Classifications | Detections | Segments | Poses,
    *,
    labels: list[str] | np.ndarray | None = None,
    max_items: int = 5,
) -> str:
    """Return a short human-readable summary of model outputs for --debug."""
    if isinstance(results, Classifications):
        n = len(results)
        if n == 0:
            return "Classifications: 0"
        order = np.argsort(results.confidence)[::-1][:max_items]
        parts: list[str] = []
        for i in order:
            cid = int(results.class_id[i])
            conf = float(results.confidence[i])
            name = None
            if labels is not None and 0 <= cid < len(labels):
                name = str(labels[cid])
            parts.append(f"{cid}{'(' + name + ')' if name else ''}={conf:.3f}")
        more = "" if n <= max_items else f" (+{n - max_items} more)"
        return f"Classifications: {n} | " + ", ".join(parts) + more

    if isinstance(results, Detections):
        n = len(results)
        if n == 0:
            return "Detections: 0"
        order = np.argsort(results.confidence)[::-1][:max_items]
        lines = [f"Detections: {n}"]
        for i in order:
            cid = int(results.class_id[i])
            conf = float(results.confidence[i])
            bbox = results.bbox[i].tolist() if results.bbox is not None else None
            name = None
            if labels is not None and 0 <= cid < len(labels):
                name = str(labels[cid])
            lines.append(f"  - cls={cid}{' (' + name + ')' if name else ''} conf={conf:.3f} bbox={bbox}")
        if n > max_items:
            lines.append(f"  ... (+{n - max_items} more)")
        return "\n".join(lines)

    if isinstance(results, Segments):
        if results.mask is None or getattr(results.mask, "size", 0) == 0:
            return "Segments: empty"
        idx = getattr(results, "indices", [])
        idx_str = ", ".join(map(str, idx[:max_items]))
        more = "" if len(idx) <= max_items else f" (+{len(idx) - max_items} more)"
        return f"Segments: mask_shape={tuple(results.mask.shape)} | ids=[{idx_str}]{more}"

    if isinstance(results, Poses):
        n = int(getattr(results, "n_detections", 0))
        if n == 0:
            return "Poses: 0"
        conf = getattr(results, "confidence", np.empty((0,)))
        top = float(np.max(conf)) if getattr(conf, "size", 0) else 0.0
        return f"Poses: {n} | top_conf={top:.3f}"

    return str(results)


def annotate_fn(
    annotator: Annotator,
    image: np.ndarray,
    image_type: IMAGE_TYPE,
    image_color_format: COLOR_FORMAT,
    results: Classifications | Detections | Segments | Poses,
    roi: ROI,
    labels: list[str] | None,
    *,
    detection_threshold: float | None = None,
) -> Frame:
    """Annotate frame with model results.

    Thresholds only affect visualization (displayed images), not evaluation.
    Defaults: detections 0.55, poses 0.15 unless overridden by caller.
    """
    h, w, c = image.shape
    frame = Frame(
        timestamp=datetime.now(),
        image=image,
        image_type=image_type,
        color_format=image_color_format,
        width=w,
        height=h,
        channels=c,
        detections=results,
        roi=roi,
        new_detection=True,
        fps=0,
        dps=0,
        frame_count=0,
    )

    if isinstance(results, Classifications):
        for i in range(min(3, len(results.class_id))):
            cid = results.class_id[i]
            label = labels[cid] if labels is not None else f"class_{cid}"
            text = f"{i + 1}. {label}: {results.confidence[i]:.2f}"
            cv2.putText(
                frame.image,
                text,
                (50, 30 + 40 * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 0, 100),
                2,
            )
    elif isinstance(results, Detections):
        thresh = 0.55 if detection_threshold is None else detection_threshold
        results = results[results.confidence > thresh]
        annotator.annotate_boxes(frame, results, labels=labels, alpha=0.3, corner_radius=10)
    elif isinstance(results, Segments):
        annotator.annotate_segments(frame, results)
    elif isinstance(results, Poses):
        thresh = 0.15 if detection_threshold is None else detection_threshold
        results = results[results.confidence > thresh]
        annotator.annotate_keypoints(frame, results, keypoint_score_threshold=0.10)

    return frame


def maybe_display_annotations(
    annotator: Annotator,
    sample,
    it_image,
    model,
    detections,
    roi: ROI,
    args,
    *,
    sleep_after_display: float = 0,
) -> None:
    """If args.display_image or args.display_input_tensor, annotate and display (optionally sleep after)."""
    if getattr(args, "display_image", False):
        frame = annotate_fn(
            annotator=annotator,
            image=sample.image.copy(),
            image_type=IMAGE_TYPE.SOURCE,
            image_color_format=sample.image_color_format,
            results=detections,
            roi=roi,
            labels=model.labels if hasattr(model, "labels") else None,
            detection_threshold=getattr(args, "annot_threshold", 0.55),
        )
        frame.display()
        if sleep_after_display > 0:
            time.sleep(sleep_after_display)

    if getattr(args, "display_input_tensor", False):
        frame = annotate_fn(
            annotator=annotator,
            image=it_image,
            image_type=IMAGE_TYPE.INPUT_TENSOR,
            image_color_format=model.color_format,
            results=detections,
            roi=ROI(left=0, top=0, width=1, height=1),
            labels=model.labels if hasattr(model, "labels") else None,
            detection_threshold=getattr(args, "annot_threshold", 0.55),
        )
        frame.display()
        if sleep_after_display > 0:
            time.sleep(sleep_after_display)


def print_settings(args, model, dataset, evaluator, mode: str = "Evaluation", model_uri: str | None = None):
    """Print evaluation settings. For interpreter mode, pass model_uri (path to model, shown as model_file)."""
    print("=" * 80)
    print(f"\033[1;36m[{mode}] Settings\033[0m")
    print(f"  \033[1mmodel\033[0m: {args.model} (\033[1;32m{model.__class__.__name__}\033[0m)")
    if model_uri is not None:
        print(f"  \033[1mmodel_file\033[0m: {model_uri}")
    elif hasattr(model, "model_file") and model.model_file:
        print(f"  \033[1mmodel_file\033[0m: {model.model_file}")
    print(f"  \033[1mtask\033[0m: {model.task_type}")
    print(f"  \033[1mdataset\033[0m: {dataset.images_dir}")
    print(f"  \033[1mground_truth\033[0m: {evaluator.ground_truth}")
    for k, v in vars(args).items():
        if k not in ("model", "dataset", "ground_truth", "model_file"):  # Skip fields already printed
            if v is not None:
                print(f"  \033[1m{k}\033[0m: {v}")
    print("=" * 80)


def print_progress(
    start_time: float,
    idx: int,
    total_samples: int,
    *,
    batch_done: int | None = None,
    batch_remaining: int | None = None,
):
    """
    Print progress information including samples processed and estimated time remaining.

    Args:
        start_time: Start time from time.time()
        idx: Current sample index (0-based); total done = idx + 1
        total_samples: Total number of samples (for display).
        batch_done: When resuming, number of samples processed in this run (for time estimate).
        batch_remaining: When resuming, number of samples left in this run (for time estimate).
    """
    done_total = idx + 1
    elapsed_time = time.time() - start_time

    if batch_done is not None and batch_remaining is not None and batch_done > 0:
        # Resume: base time estimate only on this run
        avg_time_per_sample = elapsed_time / batch_done
        estimated_time_left = avg_time_per_sample * batch_remaining
    else:
        avg_time_per_sample = elapsed_time / done_total if done_total > 0 else 0
        estimated_time_left = avg_time_per_sample * (total_samples - done_total)

    time_left_min = int(estimated_time_left // 60)
    time_left_sec = int(estimated_time_left % 60)
    percent = done_total / total_samples * 100 if total_samples > 0 else 0

    print(
        f"Processed {done_total} / {total_samples} samples ({percent:.1f}%) "
        f"- Estimated time left: {time_left_min}m {time_left_sec}s"
    )


# -----------------------------------------------------------------------------
# Checkpoint / resume (ID-based, strict config)
# -----------------------------------------------------------------------------
#
# Evaluation pickle format (standardized for evaluators):
#   The pickle file is always a dict with keys:
#     - "meta": dict of run config (mode, model, dataset paths, prep, etc.) for resume checks.
#     - "processed_ids": list of sample IDs that have been processed.
#     - "samples": list of EvaluationSample used for evaluation and visualization.
#   Evaluators should load the file and use data["samples"]; "meta" and "processed_ids"
#   are used for resume and compatibility. Legacy files (raw list of samples) are
#   accepted on load and normalized to this shape (meta=None, processed_ids=[]).

CHECKPOINT_VERSION = 1
META_KEYS = (
    "version",
    "mode",
    "model_alias",
    "model_class",
    "task_type",
    "dataset_path",
    "ground_truth_path",
    "prep",
    "extra_args",
)
# Keys in extra_args that are ignored for strict resume (not relevant to evaluation)
EXTRA_ARGS_IGNORED_FOR_RESUME = ("frame_rate",)
EVALUATION_PICKLE_KEYS = ("meta", "processed_ids", "samples")


def make_checkpoint_meta(
    mode: str,
    dataset_path: str | Path,
    ground_truth_path: str | Path | None,
    args,
    model=None,
) -> dict:
    """Build meta dict for strict resume comparison. Paths are normalized."""
    dataset_path = str(Path(dataset_path).resolve())
    ground_truth_path = str(Path(ground_truth_path).resolve()) if ground_truth_path else None
    extra = {}
    for k in ("frame_rate", "endpoint"):
        if hasattr(args, k):
            extra[k] = getattr(args, k)
    return {
        "version": CHECKPOINT_VERSION,
        "mode": mode,
        "model_alias": getattr(args, "model", None),
        "model_class": getattr(model, "__class__", type(None)).__name__ if model else None,
        "task_type": str(getattr(model, "task_type", None)) if model else None,
        "dataset_path": dataset_path,
        "ground_truth_path": ground_truth_path,
        "prep": getattr(args, "prep", None),
        "extra_args": extra,
    }


def load_checkpoint(path: str | Path) -> dict | None:
    """Load checkpoint from pickle file (standardized format). Returns None if file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Invalid or corrupt checkpoint file {path}: {e}") from e
    if not isinstance(data, dict) or "meta" not in data or "processed_ids" not in data or "samples" not in data:
        raise ValueError(f"Invalid checkpoint format in {path}")
    return data


def load_evaluation_pickle(path: str | Path) -> dict | None:
    """Load evaluation pickle in standardized format for evaluators.
    Returns None if file does not exist. Accepts both standardized dict
    (meta, processed_ids, samples) and legacy list-of-samples; legacy is
    normalized to {\"meta\": None, \"processed_ids\": [], \"samples\": list}.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Invalid or corrupt pickle file {path}: {e}") from e
    if isinstance(data, dict) and all(k in data for k in EVALUATION_PICKLE_KEYS):
        return data
    if isinstance(data, list):
        return {"meta": None, "processed_ids": [], "samples": data}
    raise ValueError(
        f"Unsupported pickle format in {path}. "
        "Expected standardized format (dict with 'meta', 'processed_ids', 'samples') or legacy list of samples."
    )


def save_checkpoint(path: str | Path, meta: dict, processed_ids: set | list, samples: list) -> None:
    """Save evaluation results in standardized format (meta, processed_ids, samples)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "processed_ids": list(processed_ids),
        "samples": samples,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def check_meta_compatible(saved_meta: dict, current_meta: dict) -> None:
    """Strict resume: raise ValueError if any config differs (except extra_args keys in EXTRA_ARGS_IGNORED_FOR_RESUME)."""
    for k in META_KEYS:
        if k not in saved_meta or k not in current_meta:
            continue
        saved_val = saved_meta[k]
        current_val = current_meta[k]
        if k == "extra_args" and isinstance(saved_val, dict) and isinstance(current_val, dict):
            # Compare extra_args but ignore keys that don't affect evaluation (e.g. frame_rate)
            saved_filtered = {x: saved_val[x] for x in saved_val if x not in EXTRA_ARGS_IGNORED_FOR_RESUME}
            current_filtered = {x: current_val[x] for x in current_val if x not in EXTRA_ARGS_IGNORED_FOR_RESUME}
            if saved_filtered != current_filtered:
                raise ValueError(
                    f"Checkpoint config mismatch: {k!r} differs "
                    f"(checkpoint: {saved_filtered!r}, current: {current_filtered!r}). "
                    "Resume aborted (strict resume)."
                )
        elif saved_val != current_val:
            raise ValueError(
                f"Checkpoint config mismatch: {k!r} differs "
                f"(checkpoint: {saved_meta[k]!r}, current: {current_meta[k]!r}). "
                "Resume aborted (strict resume)."
            )


def filter_unprocessed(
    samples_iterable: Iterable,
    processed_ids: set,
    sample_to_id: Callable,
) -> Iterable:
    """Yield only samples whose id is not in processed_ids."""
    for s in samples_iterable:
        sid = sample_to_id(s)
        if sid not in processed_ids:
            yield s


@dataclass(frozen=True)
class ResumeContext:
    """Context for resume and periodic checkpointing passed into run_* functions."""

    work_list: list | None
    total_samples: int
    initial_samples: list | None
    initial_processed_ids: set | None
    checkpoint_path: str | None
    meta: dict | None
    checkpoint_interval: int
    sample_to_id: Callable | None


def prepare_resume_state(
    mode: str,
    dataset,
    evaluator,
    model,
    args,
    sample_to_id: Callable,
) -> tuple[ResumeContext, list, set, dict]:
    """Build meta, optionally load checkpoint for resume, and return ResumeContext plus initial state.
    Returns (resume_ctx, all_samples, processed_ids, meta).
    """
    meta = make_checkpoint_meta(
        mode=mode,
        dataset_path=dataset.images_dir,
        ground_truth_path=evaluator.ground_truth,
        args=args,
        model=model,
    )
    processed_ids: set = set()
    all_samples: list = []

    if getattr(args, "resume", False) and getattr(args, "pickle_file", None):
        cp = load_checkpoint(args.pickle_file)
        if cp is not None:
            check_meta_compatible(cp["meta"], meta)
            processed_ids = set(cp["processed_ids"])
            all_samples = list(cp["samples"])

    # Only build work_list when resuming (processed_ids/all_samples non-empty);
    # otherwise keep None so callers iterate over dataset directly (avoids eager
    # full-dataset materialization for large datasets).
    work_list = (
        list(filter_unprocessed(dataset, processed_ids, sample_to_id)) if (processed_ids or all_samples) else None
    )
    checkpoint_interval = max(0, int(getattr(args, "checkpoint_interval", 0)))
    pickle_file = getattr(args, "pickle_file", None)

    ctx = ResumeContext(
        work_list=work_list,
        total_samples=len(dataset),
        initial_samples=all_samples if all_samples else None,
        initial_processed_ids=processed_ids if processed_ids else None,
        checkpoint_path=pickle_file if checkpoint_interval > 0 else None,
        meta=meta if checkpoint_interval > 0 else None,
        checkpoint_interval=checkpoint_interval,
        sample_to_id=sample_to_id if checkpoint_interval > 0 else None,
    )
    return (ctx, all_samples, processed_ids, meta)


def add_common_arguments(parser):
    """Add common command-line arguments shared by both evaluators."""
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model alias (zoo). Omit when using --model-config.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Python config file that defines a custom model setup: "
            "MODEL_PATH, MODEL_CLASS, dataset logic, etc. Use for custom-trained models. "
            "Overrides -m/--model."
        ),
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the model. AiCamera: packerOut.zip or .rpk. "
            "Interpreter: .keras or .onnx (container path or host path under --models-host). "
            "With --model-config: overrides MODEL_PATH (aicamera) or MODEL_URI (interpreter). "
            "With -m/--model: required for interpreter; for aicamera use your own file instead of zoo default."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output during evaluation (shows detailed predictions)",
    )
    parser.add_argument(
        "--annot-threshold",
        type=float,
        default=0.55,
        metavar="T",
        help=(
            "Confidence threshold for annotated images (display and saved). "
            "Used for classification, detection and pose visualization. "
            "For pose visualization, try setting a static threshold of 0.15."
            "Evaluation uses unfiltered predictions. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--display_image",
        action="store_true",
        help="Display original sample image.",
    )
    parser.add_argument(
        "--display_input_tensor",
        action="store_true",
        help="Display input tensor image. (Can be larger or smaller then original sample image depending on model preprocessing)",
    )
    parser.add_argument(
        "-p",
        "--pickle-file",
        type=str,
        default="tmp/all_samples.pkl",
        help="Path to pickle file for loading/saving samples (default: `tmp/all_samples.pkl`)",
    )
    parser.add_argument(
        "-l",
        "--load-samples",
        action="store_true",
        help="Load samples from pickle file and run evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset directory with images. For classification, the directory should be organized in the format: <class_id>/<image>",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to the ground truth file/directory. For classification the directory should be organized in the format: <class_id>/<image>. For detection and pose estimation it expects a file in COCO format, and segmentation it expects a directory with png masks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (pickle file). Fails if config differs (strict).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        metavar="N",
        help="Save checkpoint every N samples (0 = only at end). Ignored when --load-samples. Default: %(default)s.",
    )
    parser.add_argument(
        "--annotated-output-dir",
        type=str,
        default="tmp/annotated_images",
        metavar="DIR",
        help="Output directory for saved annotated images (default: %(default)s)",
    )
