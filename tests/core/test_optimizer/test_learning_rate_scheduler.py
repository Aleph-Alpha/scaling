import pytest

from scaling.core import (
    LearningRateDecayStyle,
    LearningRateScheduler,
    LearningRateSchedulerConfig,
)


@pytest.mark.parametrize(
    "scheduler_config",
    [
        {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.001,
            "learning_rate_decay_style": LearningRateDecayStyle.LINEAR,
            "learning_rate_decay_iters": 10,
            "learning_rate_warmup_steps": 5,
        },
        {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.001,
            "learning_rate_decay_style": LearningRateDecayStyle.CONSTANT,
            "learning_rate_decay_iters": 10,
            "learning_rate_warmup_steps": 5,
        },
        {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.001,
            "learning_rate_decay_style": LearningRateDecayStyle.COSINE,
            "learning_rate_decay_iters": 10,
            "learning_rate_warmup_steps": 5,
        },
        {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.001,
            "learning_rate_decay_style": LearningRateDecayStyle.LINEAR,
            "learning_rate_decay_iters": 10,
            "learning_rate_warmup_steps": 0,
        },
        {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.001,
            "learning_rate_decay_style": LearningRateDecayStyle.CONSTANT,
            "learning_rate_decay_iters": 10,
            "learning_rate_warmup_steps": 0,
        },
        {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.001,
            "learning_rate_decay_style": LearningRateDecayStyle.COSINE,
            "learning_rate_decay_iters": 10,
            "learning_rate_warmup_steps": 0,
        },
    ],
)
@pytest.mark.parametrize("train_iters", [4, 5, 8, 10, 15, 20])
def test_should_schedule_learning_rate(scheduler_config: dict, train_iters: int):
    config = LearningRateSchedulerConfig(**scheduler_config)
    scheduler = LearningRateScheduler(config=config)

    previous_lr = config.learning_rate if config.learning_rate_warmup_steps == 0 else 0.0
    for step_index in range(1, train_iters + 1):
        lr = scheduler.get_lr(step_index=step_index)

        if config.learning_rate_decay_style == LearningRateDecayStyle.CONSTANT:
            if step_index < config.learning_rate_warmup_steps:
                assert lr > previous_lr, (
                    f"during warmup learning rate does not increase; "
                    f"style {config.learning_rate_decay_style} step {step_index} lr {lr} previous_lr {previous_lr} "
                    f"config {scheduler_config}"
                )
            elif step_index > config.learning_rate_warmup_steps:
                assert lr == config.learning_rate, (
                    f"constant scheduler does not remain at constant lr; "
                    f"style {config.learning_rate_decay_style} step {step_index} lr {lr} previous_lr {previous_lr} "
                    f"config {scheduler_config}"
                )
        else:
            if step_index < config.learning_rate_warmup_steps:
                assert lr > previous_lr, (
                    f"during warmup learning rate does not increase; "
                    f"style {config.learning_rate_decay_style} step {step_index} lr {lr} previous_lr {previous_lr} "
                    f"config {scheduler_config}"
                )
            elif config.learning_rate_warmup_steps < step_index < config.learning_rate_decay_iters:
                assert lr < previous_lr, (
                    f"after warmup learning rate does not decrease; "
                    f"style {config.learning_rate_decay_style} step {step_index} lr {lr} previous_lr {previous_lr} "
                    f"config {scheduler_config}"
                )
            elif step_index > config.learning_rate_decay_iters:
                assert lr == config.learning_rate_minimum, (
                    f"after decay learning rate is not at minimum; "
                    f"style {config.learning_rate_decay_style} step {step_index} lr {lr} previous_lr {previous_lr} "
                    f"config {scheduler_config}"
                )

        previous_lr = lr
