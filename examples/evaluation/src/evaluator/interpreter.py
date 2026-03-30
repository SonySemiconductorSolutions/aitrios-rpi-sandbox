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

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from functools import partial

import modlib.models.zoo as zoo
from modlib.apps import Annotator
from modlib.devices import Dataset, Device, InterpreterClient
from modlib.devices.imx500 import isp
from modlib.models import Model
from modlib.models.evals import EvaluationSample

from .common import (
    ResumeContext,
    add_common_arguments,
    build_model_dataset_evaluator_from_file,
    format_debug_results,
    get_custom_model_uri,
    get_dataset,
    get_evaluator,
    host_path_to_container_uri,
    load_evaluation_pickle,
    maybe_display_annotations,
    prepare_resume_state,
    print_progress,
    print_settings,
    save_checkpoint,
)

# alias -> zoo model class name (for -m/--model). Path to the model file is always from --model-file (or host path under --models-host).
MODEL_ALIASES: dict[str, str] = {
    # Detection
    "ssd": "SSDMobileNetV2FPNLite320x320",
    "nano": "NanoDetPlus416x416",
    "effdet": "EfficientDetLite0",
    "yolo11": "YOLO11n",
    # Segmentation
    "deep": "DeepLabV3Plus",
    # Classification
    "effb0": "EfficientNetB0",
    "efflite0": "EfficientNetLite0",
    "effv2b0": "EfficientNetV2B0",
    "effv2b1": "EfficientNetV2B1",
    "effv2b2": "EfficientNetV2B2",
    "mobnet": "MobileNetV2",
    "resnet": "ResNet18",
    "mnasnet": "MNASNet1_0",
    "squeeze": "SqueezeNet1_0",
    "shuffle": "ShuffleNetV2X1_5",
    "regnetx": "RegNetX002",
    "regnety2": "RegNetY002",
    "regnety4": "RegNetY004",
    "mvitxs": "MobileViTXS",
    "mvitxxs": "MobileViTXXS",
    # Pose
    "hrnet": "HigherHRNet",
}

MODELS = {alias: getattr(zoo, name) for alias, name in MODEL_ALIASES.items()}

# Model extension -> default input format (onnx typically NCHW, keras NHWC)
_EXT_TO_FORMAT = {"onnx": "nchw", "keras": "nhwc"}


def _resolve_input_format(
    model_uri: str,
    zoo_default_uri: str | None,
    args,
    *,
    user_specified_input_format: bool = False,
) -> None:
    """Set args.input_format: use value if user passed --input-format; else infer from model_uri extension."""
    if user_specified_input_format:
        return
    uri_lower = model_uri.lower()
    ext = uri_lower.split(".")[-1] if "." in uri_lower else ""
    inferred = _EXT_TO_FORMAT.get(ext)

    if inferred is None:
        args.input_format = args.input_format or "nhwc"
        return

    prev = args.input_format
    args.input_format = inferred
    if prev == inferred:
        return
    zoo_ext = None
    if zoo_default_uri:
        zoo_lower = zoo_default_uri.lower()
        zoo_ext = zoo_lower.split(".")[-1] if "." in zoo_lower else ""
    if zoo_ext and ext != zoo_ext:
        print(
            f"Note: model_uri ends with .{ext} (zoo default .{zoo_ext}); using --input-format {inferred}",
            flush=True,
        )
    else:
        print(f"Note: model_uri ends with .{ext}; using --input-format {inferred}", flush=True)


def check_interpreter_reachable(endpoint: str, timeout: float = 5.0) -> None:
    """Verify the interpreter service at endpoint is up and responding. Exits with message on failure."""
    url = endpoint.rstrip("/") + "/"
    try:
        req = urllib.request.Request(url, method="GET")
        urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError:
        pass  # any HTTP response means the service is up
    except (urllib.error.URLError, OSError) as e:
        cause = getattr(e, "reason", e) or e
        err_msg = str(cause) if cause else type(e).__name__
        print(
            f"Interpreter service at {endpoint!r} is not responding ({err_msg}). "
            "Ensure the interpreter container is running and reachable.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_inference(
    model: Model,
    device: Device,
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    resume_ctx: ResumeContext | None = None,
) -> list[EvaluationSample]:
    """Run inference evaluation using InterpreterClient.
    When resume_ctx is provided, iterates only over work_list and extends initial_samples,
    and optionally saves checkpoint every checkpoint_interval samples.
    """
    if resume_ctx is None:
        work_list = None
        total_samples = len(dataset)
        initial_samples = None
        initial_processed_ids = None
        checkpoint_path = None
        meta = None
        checkpoint_interval = 0
        sample_to_id = None
    else:
        work_list = resume_ctx.work_list
        total_samples = resume_ctx.total_samples
        initial_samples = resume_ctx.initial_samples
        initial_processed_ids = resume_ctx.initial_processed_ids
        checkpoint_path = resume_ctx.checkpoint_path
        meta = resume_ctx.meta
        checkpoint_interval = resume_ctx.checkpoint_interval
        sample_to_id = resume_ctx.sample_to_id

    annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)
    all_samples: list[EvaluationSample] = list(initial_samples) if initial_samples is not None else []
    processed_ids: set = set(initial_processed_ids) if initial_processed_ids is not None else set()

    to_iterate = work_list if work_list is not None else dataset
    n_total = total_samples
    initial_count = len(all_samples)

    with device:
        start_time = time.time()
        for idx, sample in enumerate(to_iterate):
            # 1. Prepare input tensors
            if args.prep == "model":
                resize_fn = None
            elif args.prep == "like_isp":
                resize_fn = partial(isp.isp_resize_and_crop, model=model)
            else:
                raise ValueError(f"Invalid preprocessing method: {args.prep}")

            it_image, it, roi = model.pre_process(
                image=sample.image,
                src_color_format=sample.image_color_format,
                resize_fn=resize_fn,
            )
            # pre_process returns the tensor in the layout expected by the model (NCHW/NHWC).

            # 2. Run inference
            detections = device.infer(it)

            # Debug output (if enabled)
            if getattr(args, "debug", False):
                sid = getattr(sample, "image_id", None)
                spath = getattr(sample, "image_path", None)
                print(
                    f"[debug] sample={initial_count + idx + 1}/{n_total} image_id={sid} path={spath}",
                    flush=True,
                )
                print(
                    format_debug_results(
                        detections,
                        labels=model.labels if hasattr(model, "labels") else None,
                    ),
                    flush=True,
                )

            # 3. Store Evaluation Sample
            all_samples.append(
                EvaluationSample(
                    roi=roi,
                    detections=detections,
                    dataset_sample=sample,
                )
            )
            if sample_to_id is not None:
                processed_ids.add(sample_to_id(sample))

            # 4. Print progress (when resuming, time estimate uses only this run's batch)
            if work_list is not None:
                print_progress(
                    start_time,
                    initial_count + idx,
                    n_total,
                    batch_done=idx + 1,
                    batch_remaining=len(to_iterate) - (idx + 1),
                )
            else:
                print_progress(start_time, initial_count + idx, n_total)

            # 5. Checkpoint
            if (
                checkpoint_path
                and meta
                and checkpoint_interval > 0
                and sample_to_id is not None
                and (idx + 1) % checkpoint_interval == 0
            ):
                save_checkpoint(checkpoint_path, meta, processed_ids, all_samples)
                print(
                    f"Checkpoint saved (interval {checkpoint_interval}, {initial_count + idx + 1} samples)",
                    flush=True,
                )

            # 6. Annotation
            maybe_display_annotations(
                annotator,
                sample,
                it_image,
                model,
                detections,
                roi,
                args,
                sleep_after_display=1,
            )

    return all_samples


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for InterpreterClient evaluator."""
    parser = argparse.ArgumentParser(
        description="Model inference and evaluation for multiple CV tasks using InterpreterClient (classification, object detection, segmentation & pose estimation)"
    )

    add_common_arguments(parser)

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="Interpreter client endpoint URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--prep",
        choices=["model", "like_isp"],
        default="like_isp",
        help=(
            "Preprocessing method for input tensor: "
            "'model' uses the model's own pre-processing; "
            "'like_isp' uses ISP-like resize/crop before the model's pre-processing. "
            "Default: '%(default)s'."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["nhwc", "nchw"],
        default=None,
        help=(
            "Input tensor layout. "
            "'nhwc' (batch, height, width, channels) for .keras. "
            "'nchw' (batch, channels, height, width) for .onnx. "
            "If omitted, inferred from model file extension (.keras -> nhwc, .onnx -> nchw). Default: infer from file."
        ),
    )
    parser.add_argument(
        "--models-host",
        type=str,
        default="~/models",
        metavar="DIR",
        help="Host directory under which --model-file can be a path; translated to --models-container. Default: %(default)s.",
    )
    parser.add_argument(
        "--models-container",
        type=str,
        default="/models",
        metavar="DIR",
        help="Container path where host models are mounted (used when --model-file is under --models-host). Default: %(default)s.",
    )
    parser.add_argument(
        "--docker-container",
        type=str,
        default="interpreter",
        metavar="NAME",
        help="When model load fails, list models dir in this container. Default: %(default)s.",
    )

    return parser


def _is_http_deploy_error(e: BaseException) -> bool:
    """Return True if the exception looks like an HTTP error from deploy (4xx/5xx)."""
    if hasattr(e, "response") and getattr(e.response, "status_code", None) is not None:
        return e.response.status_code >= 400
    err_str = str(e)
    return any(
        s in err_str
        for s in (
            "400",
            "404",
            "500",
            "Bad Request",
            "Not Found",
            "Internal Server Error",
        )
    )


def _build_deploy_failure_message(e: BaseException, args: argparse.Namespace) -> str:
    """Build the stderr message shown when interpreter deploy fails (HTTP error).
    Uses args.docker_container and args.models_container for the optional listing.
    When the server sends no response body, we have no clue why it failed — tell user to check server logs.
    """
    detail = ""
    if hasattr(e, "response") and getattr(e.response, "text", None):
        detail = " " + (e.response.text or "").strip()
    listing = ""
    container_name = getattr(args, "docker_container", None) or "interpreter"
    models_dir = getattr(args, "models_container", None) or "/models"
    try:
        out = subprocess.run(
            ["docker", "exec", container_name, "ls", "-la", models_dir],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            listing = f"\n\nContents of {models_dir} in container {container_name!r}:\n" + out.stdout.strip()
        elif out.returncode != 0 and out.stderr.strip():
            listing = f"\n\n(docker exec {container_name!r} failed: {out.stderr.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as run_err:
        listing = f"\n\n(docker exec {container_name!r} failed: {run_err})"
    host_dir = getattr(args, "models_host", None) or "~/models"
    if detail:
        hint = (
            f"\n\n\033[1;33mHint:\033[0m Pass a container path (e.g. {models_dir}/my_model/model.onnx) "
            f"or a host path under {host_dir}; host paths are translated to {models_dir}/... "
            f"(mount {host_dir} at {models_dir} in the container)."
        )
    else:
        hint = "\n\n\033[1;33mHint:\033[0m The server did not return an error message. Check the interpreter server logs for the failure reason."
    return "\n\033[1;31mInterpreter failed to load model.\033[0m" + detail + listing + hint


def main(args: argparse.Namespace | None = None):
    """Main entry point for InterpreterClient evaluator."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    # Fail fast if we will need the interpreter and the container is not up
    if not args.load_samples:
        check_interpreter_reachable(args.endpoint)

    # Resolve model, dataset, evaluator: zoo (-m) or custom (--model-config)
    if getattr(args, "model_config", None):
        model, dataset, evaluator = build_model_dataset_evaluator_from_file(
            args.model_config,
            model_path=getattr(args, "model_file", None),
            dataset_path=args.dataset,
            ground_truth_path=args.ground_truth,
        )
        model_uri = get_custom_model_uri(args.model_config) or getattr(args, "model_file", None)
        if not model_uri:
            raise ValueError(
                "With --model-config, set MODEL_URI in the config or pass --model-file "
                "(path to the .keras/.onnx model for the interpreter)."
            )
        zoo_default_uri = None
    else:
        if not args.model or args.model not in MODELS:
            raise ValueError(
                "Use -m/--model with a zoo alias or --model-config for a custom model. "
                f"Zoo aliases: {', '.join(sorted(MODELS.keys()))}"
            )
        model_uri = getattr(args, "model_file", None)
        if not model_uri:
            raise ValueError(
                "With -m/--model, pass --model-file with the path to the model "
                "(container path or host path under --models-host)."
            )
        model = MODELS[args.model]()
        dataset = get_dataset(model, args.dataset)
        evaluator = get_evaluator(model, args.ground_truth)
        zoo_default_uri = None

    # Translate host path under --models-host to container path --models-container
    container_uri = host_path_to_container_uri(
        model_uri,
        host_prefix=getattr(args, "models_host", None),
        container_prefix=getattr(args, "models_container", None),
    )
    if container_uri is not None:
        model_uri = container_uri

    user_specified_input_format = args.input_format is not None
    if not user_specified_input_format:
        args.input_format = "nhwc"
    _resolve_input_format(
        model_uri,
        zoo_default_uri,
        args,
        user_specified_input_format=user_specified_input_format,
    )

    # Print settings (model_uri = framework model path in container, not zoo RPK)
    print_settings(args, model, dataset, evaluator, mode="Model Inference", model_uri=model_uri)

    # Run inference
    if args.load_samples:
        raw = load_evaluation_pickle(args.pickle_file)
        if raw is None:
            raise FileNotFoundError(f"Pickle file not found: {args.pickle_file}")
        all_samples = raw["samples"]
    else:

        def sample_to_id(s):
            return s.image_id

        resume_ctx, all_samples, _processed_ids, meta = prepare_resume_state(
            mode="interpreter",
            dataset=dataset,
            evaluator=evaluator,
            model=model,
            args=args,
            sample_to_id=sample_to_id,
        )

        device = InterpreterClient(endpoint=args.endpoint)
        print("Interpreter server is reachable.", flush=True)
        print("Loading model on interpreter (this may take a moment)...", flush=True)

        try:
            device.deploy(
                model,
                data={
                    "model_uri": model_uri,
                    "options": {"is_quantized": True},
                },
            )
        except Exception as e:
            if _is_http_deploy_error(e):
                print(
                    _build_deploy_failure_message(e, args),
                    file=sys.stderr,
                    flush=True,
                )
                sys.exit(1)
            raise

        print("Model loaded on interpreter.", flush=True)

        all_samples = run_inference(
            model=model,
            device=device,
            dataset=dataset,
            args=args,
            resume_ctx=resume_ctx,
        )
        # Always save in standardized format for evaluators
        if args.pickle_file is not None:
            ids = {s.dataset_sample.image_id for s in all_samples}
            save_checkpoint(args.pickle_file, meta, ids, all_samples)
            print(
                f"Checkpoint saved at end ({len(all_samples)} samples)",
                flush=True,
            )

    # Evaluate on full, unfiltered detections
    evaluator.evaluate(all_samples)

    # Visualize using config threshold (detection/pose use it; others ignore if not supported)
    try:
        evaluator.visualize(
            all_samples,
            output_dir=args.annotated_output_dir,
            detection_threshold=args.annot_threshold,
        )
    except TypeError:
        evaluator.visualize(all_samples, output_dir=args.annotated_output_dir)
