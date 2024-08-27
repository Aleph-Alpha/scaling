import argparse

from context import MinimalConfig  # type: ignore # ignore types due to relative vs. absolute import issues

from scaling.core.logging import logger
from scaling.core.runner import runner_main


def get_args():
    parser = argparse.ArgumentParser(description="Transformer Training Configuration", allow_abbrev=False)

    group = parser.add_argument_group(title="Run Configuration")

    group.add_argument(
        "conf_file",
        type=str,
        help="Configuration file path.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Use the function to generate a default yaml from your config
    # MinimalConfig.save_template("minimal_config.yml")

    args = get_args()
    config = MinimalConfig.from_yaml(args.conf_file)
    logger.configure(config=config.logger, name="runner")
    logger.info("running main")
    runner_main(config.runner, payload=config.as_dict())
    print("")
