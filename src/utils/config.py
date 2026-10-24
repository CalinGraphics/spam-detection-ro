"""Load the project configuration and resolve paths relative to the project root."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml and turn every entry under `paths` into an absolute Path."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["paths"] = {
        name: PROJECT_ROOT / relative for name, relative in config["paths"].items()
    }
    return config


if __name__ == "__main__":
    cfg = load_config()
    for name, p in cfg["paths"].items():
        print(f"{name:<10} {p}")
