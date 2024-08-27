from pathlib import Path

from scaling.core.runner import LaunchConfig

if __name__ == "__main__":
    config = LaunchConfig.from_launcher_args()
    assert config.payload is not None
    cache_dir = Path(config.payload["cache_dir"])
    with open(cache_dir / f"process_{config.global_rank}.json", "w") as f:
        f.write(config.json())
