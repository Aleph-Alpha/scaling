from pathlib import Path

import pytest
import torch

from scaling.core import (
    CoreParameterMeta,
    LearningRateDecayStyle,
    LearningRateSchedulerConfig,
    Optimizer,
    OptimizerConfig,
    OptimizerParamGroup,
    OptimizerParamGroupConfig,
    Topology,
    TopologyConfig,
)
from tests.core.utils import assert_nested_dicts_equal


def get_topology():
    return Topology(
        config=TopologyConfig(
            global_rank=0,
            model_parallel_size=1,
            data_parallel_size=1,
            pipe_parallel_size=1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        )
    )


@pytest.mark.short
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adamw(tmp_path: Path, dtype):
    """
    tests the life cycle of an adamw optimizer
    """

    # instantiate small model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(42, 42, bias=True)
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear1.weight,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear1.weight",
            )
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear1.bias,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear1.bias",
            )
            self.linear2 = torch.nn.Linear(42, 42, bias=True)
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear2.weight,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear2.weight",
            )
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear2.bias,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear2.bias",
            )

        def forward(self):
            torch.manual_seed(42)
            x = torch.randn((2, 42), dtype=dtype)
            x = self.linear1(x)
            x = self.linear2(x)

            return x.sum()

    model = Model().to(dtype)

    # Remember parameter values
    linear1_weight = model.linear1.weight.detach().clone()
    linear1_bias = model.linear1.bias.detach().clone()
    linear2_weight = model.linear2.weight.detach().clone()
    linear2_bias = model.linear2.bias.detach().clone()

    # instantiate
    config = OptimizerConfig(method="adamw", beta1=0.9, beta2=0.99, eps=1e-8, gradient_clipping=1.0)
    parameter_groups = [
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, p.core_parameter_meta)  # type: ignore
                for n, p in model.named_parameters()
                if n.endswith("weight")
            ],
            config=OptimizerParamGroupConfig(
                weight_decay=1.0,
                learning_rate_scheduler=LearningRateSchedulerConfig(
                    learning_rate=0.1,
                    learning_rate_minimum=0.0,
                    learning_rate_decay_style=LearningRateDecayStyle.COSINE,
                    learning_rate_warmup_steps=2,
                    learning_rate_decay_iters=10,
                ),
            ),
        ),
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, p.core_parameter_meta)  # type: ignore
                for n, p in model.named_parameters()
                if n.endswith("bias")
            ],
            config=OptimizerParamGroupConfig(
                **{
                    "weight_decay": 0.0,
                    "learning_rate_scheduler": {
                        "learning_rate": 0.1,
                        "learning_rate_minimum": 0.0,
                        "learning_rate_decay_style": "cosine",
                        "learning_rate_warmup_steps": 2,
                        "learning_rate_decay_iters": 10,
                    },
                }
            ),
        ),
    ]
    optimizer = Optimizer(config=config, parameter_groups=parameter_groups, topology=get_topology())

    # train a little
    for _ in range(10):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.log_state()

    # save and load
    optimizer.save_checkpoint(tmp_path)

    optimizer_new = Optimizer(config=config, parameter_groups=parameter_groups, topology=get_topology())
    optimizer_new.load_checkpoint(tmp_path)

    state_dict = optimizer.state_dict()
    assert_nested_dicts_equal(
        state_dict,
        optimizer_new.state_dict(),
        "state_dict",
    )

    # make sure that optimizer states are in fp32
    for param_group in state_dict["optimizer"]["param_groups"]:
        for param_index in param_group["params"]:
            assert state_dict["optimizer"]["state"][param_index]["exp_avg"].dtype == torch.float32
            assert state_dict["optimizer"]["state"][param_index]["exp_avg_sq"].dtype == torch.float32

    # make sure the model has been trained
    # with gradient clipping not all parameters may be updated
    # The assumption of updating all parameters is relaxed
    assert (linear1_weight != model.linear1.weight).any()
    assert (linear1_bias != model.linear1.bias).any()
    assert (linear2_weight != model.linear2.weight).any()
    assert (linear2_bias != model.linear2.bias).any()


@pytest.mark.short
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero", [True, False])
def test_refresh_optimizer_after_model_change(dtype, zero):
    # instantiate small model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(42, 42, bias=True)
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear1.weight,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear1.weight",
            )
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear1.bias,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear1.bias",
            )
            self.linear2 = torch.nn.Linear(42, 42, bias=True)
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear2.weight,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear2.weight",
            )
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear2.bias,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear2.bias",
            )

        def forward(self):
            torch.manual_seed(42)
            x = torch.randn((2, 42), dtype=dtype).cuda()
            x = self.linear1(x)
            x = self.linear2(x)

            return x.sum()

    model = Model().to(dtype).cuda()

    # instantiate
    config = OptimizerConfig(beta1=0.9, beta2=0.99, eps=1e-8, gradient_clipping=1.0, zero=zero)
    parameter_groups = [
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, p.core_parameter_meta) for n, p in model.named_parameters() if n.endswith("weight")
            ],
            config=OptimizerParamGroupConfig(
                weight_decay=1.0,
                learning_rate_scheduler=LearningRateSchedulerConfig(
                    learning_rate=0.1,
                    learning_rate_minimum=0.0,
                    learning_rate_decay_style=LearningRateDecayStyle.COSINE,
                    learning_rate_warmup_steps=2,
                    learning_rate_decay_iters=10,
                ),
            ),
        ),
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, p.core_parameter_meta) for n, p in model.named_parameters() if n.endswith("bias")
            ],
            config=OptimizerParamGroupConfig(
                **{
                    "weight_decay": 0.0,
                    "learning_rate_scheduler": {
                        "learning_rate": 0.1,
                        "learning_rate_minimum": 0.0,
                        "learning_rate_decay_style": "cosine",
                        "learning_rate_warmup_steps": 2,
                        "learning_rate_decay_iters": 10,
                    },
                }
            ),
        ),
    ]
    optimizer = Optimizer(config=config, parameter_groups=parameter_groups, topology=get_topology())

    # make sure refresh works
    optimizer.refresh_optimizer_after_model_change()
