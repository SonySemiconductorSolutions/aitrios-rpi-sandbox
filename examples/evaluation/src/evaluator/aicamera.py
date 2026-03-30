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
import time
from functools import partial
from pathlib import Path

import modlib.models.zoo as zoo
from modlib.apps import Annotator
from modlib.devices import AiCamera, Dataset, Device
from modlib.devices.imx500 import isp
from modlib.models import MODEL_TYPE, Model
from modlib.models.evals import EvaluationSample

from .common import (
    ResumeContext,
    add_common_arguments,
    build_model_dataset_evaluator_from_file,
    format_debug_results,
    get_dataset,
    get_evaluator,
    load_evaluation_pickle,
    maybe_display_annotations,
    prepare_resume_state,
    print_progress,
    print_settings,
    save_checkpoint,
)

# TODO: this should be built from modlib classes
# alias -> model class name
MODEL_ALIASES: dict[str, str] = {
    # Detection models
    "ssd": "SSDMobileNetV2FPNLite320x320",
    "yolo8": "YOLOv8n",
    "yolo11": "YOLO11n",
    "nano": "NanoDetPlus416x416",
    "effdet": "EfficientDetLite0",
    # Segmentation models
    "deep": "DeepLabV3Plus",
    # Classification models
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
    # Pose models
    "hrnet": "HigherHRNet",
    "pose": "Posenet",
}

MODELS = {alias: getattr(zoo, model_name) for alias, model_name in MODEL_ALIASES.items()}


def run_data_injection(
    model: Model,
    device: Device,
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    resume_ctx: ResumeContext | None = None,
) -> list[EvaluationSample]:
    """Run data injection evaluation using AiCamera.
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
            # 1. Data injection - choose preprocessing method based on args.prep
            if args.prep in ["model", "like_isp"]:
                # Option 1/2: model_preprocess(apply_isp_limitations=False/True) + prepare_input_tensor_for_dsp

                # `model` -> resize_fn = None: Use model's own resize/crop/padding strategy .
                # `like_isp` -> resize_fn include ISP limitations in resizing/cropping/padding.
                resize_fn = partial(isp.isp_resize_and_crop, model=model) if args.prep == "like_isp" else None

                # Pre-processing
                it_image, it, roi = model.pre_process(
                    image=sample.image,
                    src_color_format=sample.image_color_format,
                    resize_fn=resize_fn,
                )

                dsp_input_tensor = isp.prepare_input_tensor_for_dsp(it, model)
            elif args.prep == "isp":
                # Option 3: prepare_tensor_like_isp
                dsp_input_tensor, roi = isp.prepare_tensor_like_isp(
                    img=sample.image,
                    model=model,
                    src_color_format=sample.image_color_format,
                )
                it_image = isp.isp_denormalize_input_tensor(dsp_input_tensor, model)
            else:
                raise ValueError(f"Unsupported preprocessing method: {args.prep}")

            # Inject input tensor
            detections = device.inject(dsp_input_tensor)

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

            # 2. Store Evaluation Sample
            all_samples.append(
                EvaluationSample(
                    roi=roi,
                    detections=detections,
                    dataset_sample=sample,
                )
            )
            if sample_to_id is not None:
                processed_ids.add(sample_to_id(sample))

            # 3. Print progress (when resuming, time estimate uses only this run's batch)
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

            # 4. Checkpoint
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

            # 5. Annotation
            maybe_display_annotations(
                annotator,
                sample,
                it_image,
                model,
                detections,
                roi,
                args,
            )

    return all_samples


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for AiCamera evaluator."""
    parser = argparse.ArgumentParser(
        description="Data injection evaluation for multiple CV tasks using AiCamera (classification, object detection, segmentation & pose estimation)"
    )

    add_common_arguments(parser)

    parser.add_argument(
        "--prep",
        choices=["model", "like_isp", "isp"],
        default="isp",
        help=(
            "Preprocessing method for DSP input tensor: "
            "'model' uses the model's own pre-processing (no ISP simulation), "
            "'like_isp' uses the model's pre-processing with ISP-like resize/crop, "
            "'isp' uses the ISP-style tensor path (prepare_tensor_like_isp). "
            "Default: '%(default)s'."
        ),
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=5,
        help="Frame rate for AiCamera (default: 5)",
    )
    return parser


def main(args: argparse.Namespace | None = None):
    """Main entry point for AiCamera evaluator."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    # Resolve model, dataset, evaluator: zoo (-m) or custom (--model-config)
    if getattr(args, "model_config", None):
        model, dataset, evaluator = build_model_dataset_evaluator_from_file(
            args.model_config,
            model_path=getattr(args, "model_file", None),
            dataset_path=args.dataset,
            ground_truth_path=args.ground_truth,
        )
    else:
        if not args.model or args.model not in MODELS:
            raise ValueError(
                "Use -m/--model with a zoo alias or --model-config with a custom model config. "
                f"Zoo aliases: {', '.join(sorted(MODELS.keys()))}"
            )
        model = MODELS[args.model]()
        model_path_override = getattr(args, "model_file", None)
        if model_path_override is not None:
            path = Path(model_path_override).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Model path does not exist: {path}")
            model.model_file = str(path)
            # packerOut.zip requires CONVERTED (gets packaged to RPK); .rpk uses RPK_PACKAGED
            if path.suffix.lower() == ".zip":
                model.model_type = MODEL_TYPE.CONVERTED
        dataset = get_dataset(model, args.dataset)
        evaluator = get_evaluator(model, args.ground_truth)

    # Print settings
    print_settings(args, model, dataset, evaluator, mode="Data Injection")

    # Data injection
    if args.load_samples:
        raw = load_evaluation_pickle(args.pickle_file)
        if raw is None:
            raise FileNotFoundError(f"Pickle file not found: {args.pickle_file}")
        all_samples = raw["samples"]
    else:

        def sample_to_id(s):
            return s.image_id

        resume_ctx, all_samples, _processed_ids, meta = prepare_resume_state(
            mode="aicamera",
            dataset=dataset,
            evaluator=evaluator,
            model=model,
            args=args,
            sample_to_id=sample_to_id,
        )

        device = AiCamera(
            data_injection=True,
            frame_rate=getattr(args, "frame_rate", 5),
            enable_input_tensor=False,
        )
        device.deploy(model)

        all_samples = run_data_injection(
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
