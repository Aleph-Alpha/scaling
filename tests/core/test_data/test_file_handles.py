from errno import ESTALE
from unittest import mock

import pytest

from scaling.core.data.file_handles import FileHandle, RetryableException


@pytest.mark.short
def test_retry_on_stale_file_handles():
    """
    tests that retrying on stale file handles works as expected
    """
    max_attempts = 5

    class FileMock:
        def __init__(self) -> None:
            self.attempt = 0

        def open(self, *args, **kwargs):
            self.attempt += 1
            return self

        def close(self):
            pass

        def read(self, *args, **kwargs):
            if self.attempt == max_attempts:
                return b""
            else:
                err = OSError()
                err.strerror = "Stale file handle"
                err.errno = ESTALE
                raise err

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

    fm = FileMock()
    fh = FileHandle("any filepath")

    with mock.patch("builtins.open", fm.open):
        fh.retry_operation(lambda file: file.read(), max_delay=0)


def test_retry_on_retryable_exception():
    """
    tests that retrying on RetryableException works as expected
    """
    max_attempts = 5

    class FileMock:
        def __init__(self) -> None:
            self.attempt = 0

        def open(self, *args, **kwargs):
            self.attempt += 1
            return self

        def close(self):
            pass

        def read(self, *args, **kwargs):
            if self.attempt == max_attempts:
                return b""
            else:
                raise RetryableException("Some error message")

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

    fm = FileMock()
    fh = FileHandle("any filepath")

    with mock.patch("builtins.open", fm.open):
        fh.retry_operation(lambda file: file.read(), max_delay=0)
