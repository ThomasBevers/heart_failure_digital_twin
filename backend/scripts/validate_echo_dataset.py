"""Validate EchoNet-Dynamic metadata without performing model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.heart_twin.imaging import write_echonet_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local EchoNet-Dynamic research dataset.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "echonet_dynamic")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "echonet_manifest.json")
    parser.add_argument("--sample-size", type=int, default=25)
    args = parser.parse_args()
    manifest = write_echonet_manifest(args.data_dir, args.output, args.sample_size)
    report = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({key: report[key] for key in ("usable", "file_list_rows", "video_files", "videos_with_traces", "missing_videos")}, indent=2))
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
