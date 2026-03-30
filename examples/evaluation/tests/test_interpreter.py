#
# Copyright 2026 Sony  Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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

"""Tests for interpreter-specific features (e.g. input format auto-detect, deploy failure)."""

import argparse
from unittest.mock import MagicMock, patch

from evaluator.interpreter import (
    _build_deploy_failure_message,
    _is_http_deploy_error,
    _resolve_input_format,
)


class _Args:
    def __init__(self, input_format: str = "nhwc"):
        self.input_format = input_format


def test_resolve_input_format_onnx_override_when_zoo_is_keras(capsys):
    """When model_uri is .onnx and zoo default is .keras, override to nchw and inform user."""
    args = _Args(input_format="nhwc")
    _resolve_input_format(
        model_uri="/models/mobilenet_v2/mobilenet.onnx",
        zoo_default_uri="/models/mobilenet_v2/mobilenetv2.keras",
        args=args,
    )
    assert args.input_format == "nchw"
    out = capsys.readouterr()
    assert "model_uri ends with .onnx (zoo default .keras)" in out.out
    assert "using --input-format nchw" in out.out


def test_resolve_input_format_keras_override_when_zoo_is_onnx(capsys):
    """When model_uri is .keras and zoo default is .onnx, override to nhwc and inform user."""
    args = _Args(input_format="nchw")
    _resolve_input_format(
        model_uri="/models/resnet18/resnet.keras",
        zoo_default_uri="/models/resnet18/resnet.onnx",
        args=args,
    )
    assert args.input_format == "nhwc"
    out = capsys.readouterr()
    assert "model_uri ends with .keras (zoo default .onnx)" in out.out
    assert "using --input-format nhwc" in out.out


def test_resolve_input_format_no_override_when_same_as_zoo(capsys):
    """When model_uri extension matches zoo default, do not override."""
    args = _Args(input_format="nhwc")
    _resolve_input_format(
        model_uri="/models/mobilenet_v2/mobilenetv2.keras",
        zoo_default_uri="/models/mobilenet_v2/mobilenetv2.keras",
        args=args,
    )
    assert args.input_format == "nhwc"
    out = capsys.readouterr()
    assert "Note:" not in out.out


def test_resolve_input_format_no_message_when_user_already_correct(capsys):
    """When user explicitly passed --input-format nchw, do not override from extension."""
    args = _Args(input_format="nchw")
    _resolve_input_format(
        model_uri="/models/mobilenet_v2/mobilenet.onnx",
        zoo_default_uri="/models/mobilenet_v2/mobilenet.keras",
        args=args,
        user_specified_input_format=True,
    )
    assert args.input_format == "nchw"
    out = capsys.readouterr()
    assert "Note:" not in out.out


def test_resolve_input_format_custom_config_infers_from_uri(capsys):
    """When zoo_default_uri is None (custom config), infer from model_uri extension."""
    args = _Args(input_format="nhwc")
    _resolve_input_format(
        model_uri="/models/my_model/model.onnx",
        zoo_default_uri=None,
        args=args,
    )
    assert args.input_format == "nchw"
    out = capsys.readouterr()
    assert "model_uri ends with .onnx" in out.out
    assert "zoo default" not in out.out


def test_resolve_input_format_unknown_extension_no_override():
    """When model_uri has unknown extension, do not override."""
    args = _Args(input_format="nhwc")
    _resolve_input_format(
        model_uri="/models/foo/model.xyz",
        zoo_default_uri="/models/mobilenet_v2/mobilenet.keras",
        args=args,
    )
    assert args.input_format == "nhwc"


# --- Deploy failure (_is_http_deploy_error, _build_deploy_failure_message) ---


def test_is_http_deploy_error_response_400():
    """Exception with response.status_code 400 is treated as HTTP deploy error."""
    e = MagicMock()
    e.response.status_code = 400
    assert _is_http_deploy_error(e) is True


def test_is_http_deploy_error_response_500():
    """Exception with response.status_code 500 is treated as HTTP deploy error."""
    e = MagicMock()
    e.response.status_code = 500
    assert _is_http_deploy_error(e) is True


def test_is_http_deploy_error_response_200():
    """Exception with response.status_code 200 is not treated as HTTP deploy error."""
    e = MagicMock()
    e.response.status_code = 200
    assert _is_http_deploy_error(e) is False


def test_is_http_deploy_error_no_response_with_400_in_str():
    """Exception without .response but with '400' in str(e) is treated as HTTP error."""
    e = ValueError("Status: 400, Reason: Bad Request")
    assert _is_http_deploy_error(e) is True


def test_is_http_deploy_error_plain_exception():
    """Plain exception without response or status in message is not treated as HTTP error."""
    e = ValueError("something went wrong")
    assert _is_http_deploy_error(e) is False


def test_build_deploy_failure_message_includes_hint_and_detail():
    """Deploy failure message includes 'Interpreter failed', Hint, and optional response text."""
    e = MagicMock()
    e.response.text = " Model path not found "
    args = argparse.Namespace(docker_container="interpreter")
    with patch("evaluator.interpreter.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        msg = _build_deploy_failure_message(e, args)
    assert "\033[1;31mInterpreter failed to load model.\033[0m" in msg
    assert "Model path not found" in msg
    assert "Hint:" in msg
    assert "container path" in msg and "~/models" in msg


def test_build_deploy_failure_message_docker_listing():
    """When docker exec succeeds, message includes 'Contents of /models'."""
    e = MagicMock()
    e.response.text = None
    args = argparse.Namespace(docker_container="my_container")
    with patch("evaluator.interpreter.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="total 0\ndrwxr-xr-x 2 root root 4096 Jan 1 00:00 .",
            stderr="",
        )
        msg = _build_deploy_failure_message(e, args)
    assert "Contents of /models in container 'my_container':" in msg
    assert "total 0" in msg


def test_build_deploy_failure_message_docker_fails():
    """When docker exec raises (e.g. docker not found), message includes failure note."""
    e = MagicMock()
    e.response.text = None
    args = argparse.Namespace(docker_container="interpreter")
    with patch("evaluator.interpreter.subprocess.run") as run:
        run.side_effect = FileNotFoundError("docker not found")
        msg = _build_deploy_failure_message(e, args)
    assert "(docker exec 'interpreter' failed: docker not found)" in msg
    assert "Interpreter failed to load model" in msg
