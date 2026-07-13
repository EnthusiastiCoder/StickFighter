"""Persistent player save data (total coins, settings) as a small JSON file
next to the game. Purely a game-layer concern -- never touched by the sim,
the renderer, or headless training."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SAVE_PATH = Path("save.json")


@dataclass
class SaveData:
    total_coins: int = 0
    gore_enabled: bool = True


def load(path: Path = SAVE_PATH) -> SaveData:
    if not path.exists():
        return SaveData()
    try:
        raw = json.loads(path.read_text())
        return SaveData(
            total_coins=int(raw.get("total_coins", 0)),
            gore_enabled=bool(raw.get("gore_enabled", True)),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return SaveData()


def save(data: SaveData, path: Path = SAVE_PATH) -> None:
    path.write_text(json.dumps(asdict(data)))
