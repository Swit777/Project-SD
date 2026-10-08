from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from player_experience.config import load_config
from player_experience.prepare import prepare_dataset

if __name__ == "__main__":
    prepare_dataset(ROOT, load_config())
