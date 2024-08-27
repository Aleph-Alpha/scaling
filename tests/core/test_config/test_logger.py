from typing import Optional

import pytest

from scaling.core.logging import LoggerConfig, logger
from scaling.core.logging.logger_config import _check_if_in_rank


@pytest.mark.parametrize("ranks", [[], None, [0], [42], [36]])
@pytest.mark.parametrize("global_rank", [None, 0, 42])
def test_is_in_rank_like_explicit_condition(ranks: Optional[list[int]], global_rank: Optional[int]) -> None:
    """
    Checks that the check_if_in_rank function is similar to the previously very explicit condition
    """
    expected = global_rank is not None and (
        (ranks is None and global_rank == 0) or (ranks is not None and global_rank in ranks)
    )
    assert expected == _check_if_in_rank(global_rank, ranks)


def test_logger_init() -> None:
    logger.configure(LoggerConfig())
    normal_logger = repr(logger.info)
    logger.configure_determined(LoggerConfig())
    det_logger = repr(logger.info)
    logger.configure(LoggerConfig())
    normal_logger_2 = repr(logger.info)
    assert det_logger != normal_logger
    assert "bound method Logger.info of <scaling.core.logging.logging.Logger" in normal_logger
    assert "bound method Logger.info of <scaling.core.logging.logging.Logger" in normal_logger_2
    assert "bound method Logger.info of <scaling.core.logging.logging.DeterminedLogger" in det_logger
