import time
from errno import ESTALE
from pathlib import Path
from typing import IO, Any, Callable, Optional, TypeVar

V = TypeVar("V")


class RetryableException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class FileHandle:
    def __init__(self, path: Path, mode: str = "rb"):
        self._path = path
        self._mode = mode
        self._handle: Optional[IO[Any]] = None

    def retry_operation(self, func: Callable[[IO[Any]], V], max_attempts: int = 5, max_delay: int = 32) -> V:
        attempts = 0
        while True:
            try:
                if self._handle is None:
                    self._handle = open(self._path, self._mode)
                return func(self._handle)
            except Exception as e_retryable:
                if not is_retryable(e_retryable):
                    raise e_retryable

                try:
                    self.close()
                except Exception as e_close:
                    print(
                        f"Tried closing file {self._path} but it didn't work: {e_close}",
                        flush=True,
                    )

                attempts += 1
                print(
                    f"Caught retryable error for {self._path}: {e_retryable}. Attempt {attempts}/{max_attempts}.",
                    flush=True,
                )

                if attempts == max_attempts:
                    break
                else:
                    delay = min(2**attempts, max_delay)
                    time.sleep(delay)
        print(
            f"Maximum number of {max_attempts} attempts reached trying to perform file operation on {self._path}.",
            flush=True,
        )
        raise Exception(f"Stale file handle even after {attempts} retries for {self._path}.")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def is_retryable(e: Exception) -> bool:
    if isinstance(e, OSError):
        return e.errno == ESTALE
    elif isinstance(e, RetryableException):
        return True
    return False
