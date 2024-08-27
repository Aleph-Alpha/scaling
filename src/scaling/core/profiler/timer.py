import time
from typing import Optional

import torch


class SynchronizedTimer:
    """
    Timer class invoking torch.cuda.synchronize() for better multi-gpu timing.
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> None:
        torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self) -> None:
        assert self.start_time is not None, "timer has not been started and cannot be stopped"
        torch.cuda.synchronize()
        self.end_time = time.time()

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None

    def duration(self) -> float:
        assert self.start_time is not None, "timer has not been started and cannot return duration"
        assert self.end_time is not None, "timer has not been stopped and cannot return duration"

        duration = self.end_time - self.start_time

        return duration
