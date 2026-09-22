"""Download the official pretrained EchoNet-Dynamic checkpoints.

These are the authors' released weights (EF regression + LV segmentation),
used by src/echo/inference.py's EchoNetAdapter.infer(), which requires an
explicit checkpoint path and never downloads or guesses weights on its own.

Usage:
    python scripts/download_echonet_weights.py
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Model outputs belong to the project root, matching the runtime lookup in
# backend.heart_twin.echonet_integration and the repository documentation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_URLS = {
    "segmentation": "https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt",
    "ef": "https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt",
}


def download_checkpoints(destination: Path, *, force: bool = False) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in CHECKPOINT_URLS.items():
        target = destination / Path(url).name
        if target.exists() and not force:
            print(f"{name}: already present at {target}")
        else:
            print(f"{name}: downloading {url} -> {target}")
            urllib.request.urlretrieve(url, target)
        paths[name] = target
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "outputs" / "echonet_weights")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists.")
    args = parser.parse_args()
    download_checkpoints(args.destination, force=args.force)


if __name__ == "__main__":
    main()
