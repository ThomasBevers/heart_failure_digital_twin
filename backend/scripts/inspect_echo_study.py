"""Inspect the technical metadata of a local echocardiography video."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_twin.imaging import inspect_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local echocardiography video without modifying it.")
    parser.add_argument("video", type=Path, help="Path to an AVI, MP4, or MOV video")
    parser.add_argument("--view", default="Unknown", help="Confirmed clinical view, for example A4C")
    args = parser.parse_args()
    print(json.dumps(asdict(inspect_video(args.video, view=args.view)), indent=2))


if __name__ == "__main__":
    main()
