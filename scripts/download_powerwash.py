from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from player_experience.source import download_sources

if __name__ == "__main__":
    download_sources(ROOT / "data" / "raw" / "powerwash")
