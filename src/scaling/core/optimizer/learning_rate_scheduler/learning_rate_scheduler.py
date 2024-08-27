import math

from .learning_rate_scheduler_config import (
    LearningRateDecayStyle,
    LearningRateSchedulerConfig,
)


class LearningRateScheduler:
    """
    Class managing learning rate decay functions from:
    https://openreview.net/pdf?id=BJYwwY9ll pg. 4
    """

    def __init__(self, config: LearningRateSchedulerConfig):
        self.config = config

    def get_lr(self, step_index: int) -> float:
        """
        Compute the learning rate for a given step index.
        """
        # Use linear warmup for the initial part.
        if self.config.learning_rate_warmup_steps > 0 and step_index <= self.config.learning_rate_warmup_steps:
            return self.config.learning_rate * float(step_index) / float(self.config.learning_rate_warmup_steps)

        # If constant learning rate return the max after warmup
        if self.config.learning_rate_decay_style == LearningRateDecayStyle.CONSTANT:
            return self.config.learning_rate

        # For any steps larger than `self.config.learning_rate_decay_iters`, use `self.min_lr`
        if step_index > self.config.learning_rate_decay_iters:
            return self.config.learning_rate_minimum

        # Use decay styles after warmup
        # Note that to get here:
        #   self.config.learning_rate_warmup_steps < step_index <= self.config.learning_rate_decay_iters
        num_steps_no_warmup = step_index - self.config.learning_rate_warmup_steps
        decay_steps_no_warmup = self.config.learning_rate_decay_iters - self.config.learning_rate_warmup_steps
        decay_ratio = float(num_steps_no_warmup) / float(decay_steps_no_warmup)

        assert 0.0 <= decay_ratio <= 1.0
        delta_lr = self.config.learning_rate - self.config.learning_rate_minimum
        if self.config.learning_rate_decay_style == LearningRateDecayStyle.LINEAR:
            coeff = 1.0 - decay_ratio
        elif self.config.learning_rate_decay_style == LearningRateDecayStyle.COSINE:
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.config.learning_rate_minimum + coeff * delta_lr
