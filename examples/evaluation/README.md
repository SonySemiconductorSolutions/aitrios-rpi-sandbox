# Evaluator

Model evaluation for classification, object detection, segmentation, and pose estimation.

## Recommendation

Before using this library, you should:

1. **Be familiar with the [Module Application Library (modlib)](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library) version 1.4.0 or higher.**
   - This evaluator relies on examples provided by `modlib`.
   - Make sure you have cloned or downloaded the `modlib` repository.

2. **Check that the following `modlib` examples work on your setup:**
   - `examples/aicam/data_injection`:  
     - Try running the example to confirm you can inject data.
     - Follow its instructions to set up Raspberry Pi OS (Trixie or later) and verify the correct boot configuration.
   - `examples/interpreters`:  
     - Run the provided examples and confirm you can start and access the interpreter server.
     - This step ensures your environment is ready for model evaluation.

**Summary:**  
Successfully running these examples proves your system is set up correctly. Do this before proceeding with the evaluator.

## Installation

From the `tools/evaluation` directory (where `pyproject.toml` and `uv.lock` live):

```bash
uv sync
```

This installs dependencies from the lockfile.

For tests, add dev extras

```bash
uv sync --extra dev
make test
```

**Default assets (zoo):** The evaluator looks for dataset/ground-truth under `./examples/assets/` when using zoo models. To populate them, run from repo root or `tools/evaluation`:

```bash
./tools/evaluation/fetch_modlib_assets.sh
```

This clones the modlib repo to a temporary folder, runs its `examples/get_assets.py`, and copies assets into `examples/evaluation/examples/assets`.

## Usage

### AiCamera (data injection)

```bash
uv run evaluator aicamera -m yolo11
```

### Interpreter (model inference)

Requires a running interpreter container, see [modlib interpreter example](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/tree/main/examples/interpreters). In practice, clone the modlib repo. Then build images and run interpreter services using the assets in modlib folder `examples/interpreter/docker`.

> **Note:** Before running the command below, you must first export your model to ONNX format using the Ultralytics tools. For detailed instructions on exporting and preparing a YOLO model for the IMX500, see the [Ultralytics IMX500 documentation](https://docs.ultralytics.com/integrations/sony-imx500/).

```bash
# Requires onnx model from Ultralytics
uv run evaluator interpreter -m yolo11 --model-file <path-to-quantized-model-file>
```

> **Note:** Results produced by the `aicamera` and `interpreter` backends can be quite different. This is because the default exported Ultralytics YOLO11 ONNX model may not be fully optimized after conversion, particularly during the quantization step. Additional quantization and tuning can improve accuracy on the aicamera backend. Running aicamera as backend is using an optimized version from [Raspberry Pi model zoo](https://github.com/raspberrypi/imx500-models).

### Custom models

Two ways to run models:

| Flag | Purpose |
|------|---------|
| `--model-config` | Python **config file** — defines MODEL_PATH, MODEL_CLASS, dataset logic. For custom-trained models. |
| `--model-file` | Path to the **model**: AiCamera packerOut.zip or .rpk; Interpreter .keras or .onnx (container path or host path under `--models-host`). Overrides MODEL_PATH (aicamera) or MODEL_URI (interpreter) in config. **Required** with `-m` for interpreter. |

Use `--model-config` with a Python file that defines the model contract; override paths via `--model-file`, `--dataset`, `--ground-truth`. For interpreter, set `MODEL_URI` in the config or pass `--model-file`.

```bash
uv run evaluator aicamera --model-config examples/custom_od_evaluation.py
uv run evaluator interpreter --model-config examples/custom_od_evaluation.py
```

### Overriding model zoo paths

When using zoo models (`-m/--model`), `--model-file` lets you point to your own packerOut.zip or .rpk instead of the default:

```bash
# AiCamera: use an alternative .rpk or packerOut.zip (packerOut.zip is packaged to RPK automatically)
uv run evaluator aicamera -m mobnet --model-file /path/to/your/packerOut.zip
# Or, for .rpk file
uv run evaluator aicamera -m mobnet --model-file /path/to/your/network.rpk

# Interpreter: -m picks the class; --model-file is required (container path or host path under ~/models)
uv run evaluator interpreter -m mobnet --model-file /models/mobilenet_v2/your-model.keras

# When model file extension differs from zoo default (e.g. .onnx vs .keras), input format is auto-detected.
uv run evaluator interpreter -m ssd --model-file /models/ssd.onnx
uv run evaluator interpreter -m resnet --model-file /models/resnet.keras

# On model load failure, list /models in the container: --docker-container <name>
uv run evaluator interpreter -m mobnet --model-file /wrong/path.onnx --docker-container interpreter
```

### Checkpoints and resume

The evaluator periodically saves progress so you can resume evaluation if interrupted, using the `--resume` flag. Intermediate results are saved to the file specified by `--pickle-file`.

To resume, use `--resume`. Make sure that your model, dataset, ground truth, and `--prep` options are the same as before.

**Evaluation pickle format (for evaluators):** The pickle file is always saved in a standardized format: a dict with keys `meta`, `processed_ids`, and `samples`. Use `data["samples"]` for evaluation and visualization; `meta` and `processed_ids` are used for resume and compatibility checks. Load files with `evaluator.common.load_evaluation_pickle(path)` to accept both this format and legacy list-of-samples (normalized to the same dict shape).

### Listing zoo model aliases

To see which `-m/--model` aliases are available for each backend:

```bash
uv run evaluator list-models
```

### Help

```bash
uv run evaluator --help
uv run evaluator aicamera --help
uv run evaluator interpreter --help
```

## Examples

| File | Description |
|------|-------------|
| `custom_cls_evaluation.py` | Custom classification model trained on Rock/Paper/Scissors dataset. Training, quantization, and export - see [RPS tutorial](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training/tree/main/notebooks/mobilenet-rps). Model based on MobileNetV2. |
| `custom_od_evaluation.py` | Custom object detection model trained on Personal Protection Equipment dataset. Training, quantization, and export - see [PPE tutorial](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training/tree/main/notebooks/nanodet-ppe). Model based on NanoDet. |
| `object_detection_aicamera_evaluator.py` | Shows how to run evaluation from a script. |

Custom configs: use with `--model-config examples/<file>`. Scripts can use checkpoint/resume via `evaluator.aicamera` and `evaluator.common`.

## Some results
First 30 images from COCO 2017 validation dataset

`--prep like_isp`

Note: YOLOv11n IMX500 quantization was used as-is without additional optimization [Ultralytics IMX500](https://docs.ultralytics.com/integrations/sony-imx500/#python). Results improve with further tuning as illustrated with YOLOv11n*, which is an optimized model downloaded from [Raspberry Pi model zoo](https://github.com/raspberrypi/imx500-models).
| model    | backend     | AP   | AP50 | AP75 | APs  | APm  | APl  | AR1  | AR10 | AR100 | ARs  | ARm  | ARl  |
|-----------|-------------|------|------|------|------|------|------|------|------|-------|------|------|------|
| YOLOv11n  | interpreter | 0.28 | 0.44 | 0.28 | 0.16 | 0.31 | 0.40 | 0.26 | 0.30 | 0.30  | 0.16 | 0.34 | 0.42 |
| YOLOv11n  | aicamera    | 0.28 | 0.43 | 0.28 | 0.16 | 0.32 | 0.40 | 0.26 | 0.30 | 0.30  | 0.17 | 0.34 | 0.42 |
| YOLOv11n* | aicamera    | 0.36 | 0.58 | 0.34 | 0.32 | 0.39 | 0.43 | 0.28 | 0.44 | 0.46  | 0.39 | 0.50 | 0.47 |
| SSD       | interpreter | 0.35 | 0.50 | 0.36 | 0.12 | 0.39 | 0.65 | 0.32 | 0.45 | 0.46  | 0.12 | 0.49 | 0.76 |
| SSD       | aicamera    | 0.33 | 0.50 | 0.35 | 0.09 | 0.39 | 0.64 | 0.31 | 0.44 | 0.45  | 0.09 | 0.51 | 0.74 |
| NanoDet   | interpreter | 0.48 | 0.65 | 0.49 | 0.20 | 0.58 | 0.74 | 0.41 | 0.55 | 0.58  | 0.23 | 0.66 | 0.83 |
| NanoDet   | aicamera    | 0.45 | 0.62 | 0.47 | 0.18 | 0.52 | 0.74 | 0.39 | 0.54 | 0.56  | 0.21 | 0.62 | 0.83 |


## More Information

For more details about models, devices, data injection and evaluation, see the [modlib documentation](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library).