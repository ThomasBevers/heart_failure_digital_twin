import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Make backend importable
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# Make heart_twin importable
HEART_TWIN = ROOT / "heart_twin"
sys.path.insert(0, str(HEART_TWIN))

# Make vendored echonet importable
VENDOR = ROOT / "vendor" / "echonet_dynamic"
sys.path.insert(0, str(VENDOR))
