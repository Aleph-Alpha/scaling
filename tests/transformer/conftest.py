from pytest import fixture

from scaling.core.logging import LoggerConfig, logger


@fixture(autouse=True, scope="function")
def init_logger() -> None:
    logger.configure(LoggerConfig())
