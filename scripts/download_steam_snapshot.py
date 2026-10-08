from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_concept.sources import download_snapshot

if __name__ == "__main__":
    download_snapshot(ROOT / "data/raw/steam_snapshot")
