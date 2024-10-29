from pathlib import Path

from pytest import fixture

from scaling.core.logging import LoggerConfig, logger
from scaling.transformer.tokenizer import Tokenizer


@fixture(autouse=True, scope="function")
def init_logger() -> None:
    logger.configure(LoggerConfig())


@fixture(scope="session")
def unigram_02_tokenizer(path_to_files):
    return Tokenizer.from_file(str(path_to_files / "unigram_02pct_cc_v1.0_hf_converted_cleaned.json"))


@fixture(scope="session")
def path_to_files():
    return Path(__file__).parents[0] / "files"
