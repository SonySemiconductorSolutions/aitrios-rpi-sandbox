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

"""Minimal CLI smoke tests. Run with: uv run pytest tests/ -v"""

import os
import subprocess
import sys

import pytest

from evaluator import main

# Project root (examples/evaluation) and src for PYTHONPATH when running -m evaluator in subprocess
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_ENV = os.environ.copy()
_TEST_ENV["PYTHONPATH"] = os.path.join(_TEST_ROOT, "src") + os.pathsep + _TEST_ENV.get("PYTHONPATH", "")


def test_cli_help_via_subprocess_exits_zero():
    """evaluator --help via real CLI invocation (e.g. uv run evaluator --help) exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "evaluator", "--help"],
        capture_output=True,
        text=True,
        cwd=_TEST_ROOT,
        env=_TEST_ENV,
    )
    assert result.returncode == 0, result.stderr or result.stdout or "no output"
    assert "usage:" in (result.stdout or result.stderr).lower()
    assert "aicamera" in (result.stdout or result.stderr)
    assert "interpreter" in (result.stdout or result.stderr)


def test_cli_help_exits_zero():
    """evaluator --help exits with code 0."""
    sys.argv = ["evaluator", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_cli_help_when_launcher_injects_name_exits_zero():
    """evaluator --help when argv is [script, 'evaluator', '--help'] (e.g. uv run) exits 0."""
    sys.argv = ["evaluator", "evaluator", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_cli_no_args_exits_one():
    """evaluator (no args) exits with code 1."""
    sys.argv = ["evaluator"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_subcommand_help_exits_zero():
    """evaluator aicamera --help exits with code 0."""
    sys.argv = ["evaluator", "aicamera", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_cli_aicamera_help_shows_subcommand_options(capsys):
    """evaluator aicamera --help shows aicamera-specific options (e.g. -m, --dataset), not only top-level usage."""
    sys.argv = ["evaluator", "aicamera", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "-m" in out or "model" in out.lower()
    assert "dataset" in out.lower()
    assert "aicamera" in out.lower() or "evaluator aicamera" in out


def test_cli_interpreter_help_shows_subcommand_options(capsys):
    """evaluator interpreter --help shows interpreter-specific options (e.g. --endpoint)."""
    sys.argv = ["evaluator", "interpreter", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "endpoint" in out.lower()


def test_cli_aicamera_help_via_subprocess():
    """uv run evaluator aicamera --help shows aicamera help (subprocess)."""
    result = subprocess.run(
        [sys.executable, "-m", "evaluator", "aicamera", "--help"],
        capture_output=True,
        text=True,
        cwd=_TEST_ROOT,
        env=_TEST_ENV,
    )
    assert result.returncode == 0, result.stderr or result.stdout or "no output"
    out = result.stdout or result.stderr
    assert "-m" in out or "model" in out.lower()
    assert "dataset" in out.lower()


def test_cli_aicamera_help_when_launcher_injects_name_shows_subcommand_help(capsys):
    """evaluator aicamera --help when argv is [script, 'evaluator', 'aicamera', '--help'] shows aicamera help."""
    sys.argv = ["evaluator", "evaluator", "aicamera", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "dataset" in out.lower()
    assert "-m" in out or "model" in out.lower()


def test_cli_list_models(capsys):
    """evaluator list-models prints one alias per line with full model names."""
    sys.argv = ["evaluator", "list-models"]
    main()
    out = capsys.readouterr().out
    assert "Zoo model aliases" in out
    assert "aicamera" in out
    assert "interpreter" in out
    assert "ssd" in out and "SSDMobileNetV2FPNLite320x320" in out
    assert "yolo11" in out and "YOLO11n" in out
