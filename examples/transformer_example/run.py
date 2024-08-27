import argparse

from scaling.core import runner_main
from scaling.core.logging import logger
from scaling.transformer import TransformerConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = TransformerConfig.from_yaml(args.config)
    logger.configure(config=config.logger, name="runner")
    runner_main(config.runner, payload=config.as_dict())
