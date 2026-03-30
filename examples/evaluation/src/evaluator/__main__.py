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
import sys

from . import aicamera, interpreter


def create_main_parser():
    """Create the main parser for top-level help."""
    parser = argparse.ArgumentParser(
        description="Model evaluation for multiple CV tasks (classification, object detection, segmentation & pose estimation)",
        prog="evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["aicamera", "interpreter", "list-models"],
        help="Evaluation mode: aicamera, interpreter, or list-models (show zoo aliases for -m)",
    )
    return parser


def _parse_list_models_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evaluator list-models", description="List model aliases for -m/--model.")
    parser.add_argument(
        "--format",
        choices=["human", "machine"],
        default="human",
        help="human: readable table (default); machine: one alias per line for scripts",
    )
    return parser.parse_args(argv)


def _list_models(format: str = "human"):
    """Print zoo model aliases for aicamera and interpreter (for use with -m/--model).
    format: 'human' for readable table, 'machine' for one aicamera alias per line (script-friendly).
    """

    if format == "machine":
        for alias in sorted(aicamera.MODEL_ALIASES):
            print(alias)
        return

    def _rows(alias_to_name):
        return sorted(alias_to_name.items())

    aicamera_rows = _rows(aicamera.MODEL_ALIASES)  # alias -> model_name
    interpreter_rows = _rows(interpreter.MODEL_ALIASES)  # alias -> model_name

    width = max(len(alias) for alias, _ in aicamera_rows + interpreter_rows)

    print("Zoo model aliases for -m/--model:\n")
    print("  aicamera (data injection)")
    print("  " + "-" * (width + 2 + 40))
    for alias, name in aicamera_rows:
        print(f"  {alias:<{width}}  {name}")
    print("\n  interpreter (model inference)")
    print("  " + "-" * (width + 2 + 40))
    for alias, name in interpreter_rows:
        print(f"  {alias:<{width}}  {name}")
    print("\nUse: evaluator aicamera -m <alias> ...  or  evaluator interpreter -m <alias> ...")


def main():
    """Main entry point that routes to appropriate evaluator based on subcommand."""
    # Normalize argv when launcher (e.g. uv run) injects the script name as first arg so we see real args.
    # [script, "evaluator", "--help"] -> [script, "--help"]; [script, "evaluator", "aicamera", ...] -> [script, "aicamera", ...]
    if len(sys.argv) >= 2 and sys.argv[1] == "evaluator":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    # Top-level help only when there is no subcommand (e.g. "evaluator --help", not "evaluator aicamera --help")
    if len(sys.argv) == 2 and sys.argv[1] in ("--help", "-h"):
        parser = create_main_parser()
        parser.print_help()
        sys.exit(0)

    parser = create_main_parser()
    args, _unknown = parser.parse_known_args()
    command = args.command

    if command is None:
        parser.print_help()
        sys.exit(1)

    # Remove the subcommand from argv so the subcommand parser doesn't see it
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    try:
        if command == "aicamera":
            aicamera.main()
        elif command == "interpreter":
            interpreter.main()
        elif command == "list-models":
            list_args = _parse_list_models_args(original_argv[2:])
            _list_models(format=list_args.format)
        else:
            print(f"Unknown command: {command}")
            print("\nUsage: evaluator {aicamera|interpreter|list-models} [options]")
            print("Run 'evaluator --help' for more information.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)
        sys.exit(130)
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
