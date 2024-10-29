from unittest import mock

import numpy as np
import pytest

from scaling.core.data.file_dataset import retry_array_from_file
from scaling.core.data.file_handles import FileHandle


@pytest.mark.short
def test_retry_array_from_file():
    max_attempts = 5
    test_array = [1, 2]

    class FileMock:
        def __init__(self) -> None:
            self.attempt = 0

        def open(self, *args, **kwargs):
            self.attempt += 1
            return self

        def close(self):
            pass

        def np_fromfile(self, *args, **kwargs):
            if self.attempt == max_attempts:
                return test_array
            return np.array([])

        def seek(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

    fm = FileMock()
    fh = FileHandle("any filepath")

    with mock.patch("builtins.open", fm.open):
        with mock.patch("numpy.fromfile", fm.np_fromfile):
            res = retry_array_from_file(fh, "uint64", 2, 0, max_attempts=max_attempts, max_delay=0)
            assert np.array_equal(res, test_array)
