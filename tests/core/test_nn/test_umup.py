from typing import Literal

import pytest
import torch

from scaling.core.nn.attention.attention import ParallelSelfAttention
from scaling.core.nn.linear.column_parallel_linear import ColumnParallelLinear
from scaling.core.nn.linear.row_parallel_linear import RowParallelLinear
from scaling.core.nn.linear.vocab_parallel_embedding import VocabParallelEmbedding
from scaling.core.nn.masked_softmax.masked_softmax_config import MaskedSoftmaxConfig, MaskedSoftmaxKernel
from scaling.core.nn.mlp import ParallelMLP, ParallelSwiGLUMLP
from scaling.core.nn.norm.layernorm import LayerNorm
from scaling.core.nn.norm.layernorm_config import LayerNormConfig, LayerNormOptimizationType
from scaling.core.nn.norm.rms_norm import RMSNorm
from scaling.core.nn.parameter_meta import UMUP_WEIGHT_TYPE, CoreParameterMeta
from scaling.core.nn.residual import NormedResidualAdd, NormedResidualSplit
from scaling.core.nn.rotary_config import RotaryConfig
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.topology.topology import Topology
from scaling.core.topology.topology_config import TopologyConfig
from scaling.core.utils.port import find_free_port
from tests.core.utils import dist_launcher

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def is_close(x: float, x_ref: float, rtol: float = 5.0e-2) -> bool:
    return abs(x - x_ref) / x_ref < rtol


def has_scale(tensor: torch.Tensor, scale: float, rtol: float = 5.0e-2) -> bool:
    return is_close(tensor.std().item(), scale, rtol=rtol)


def get_topology(pipe_parallel_size: int, model_parallel_size: int) -> Topology:
    topology_config_dict = {
        "model_parallel_size": model_parallel_size,
        "pipe_parallel_size": pipe_parallel_size,
        "world_size": pipe_parallel_size * model_parallel_size,
        "global_rank": 0,
        "local_slot": 0,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    topology_config = TopologyConfig.from_dict(topology_config_dict)
    return Topology(config=topology_config)


def run_test_parallel_linear(
    return_dict: dict,
    model_parallel_size: int,
    in_features: int,
    out_features: int,
    batch_size: int,
    sequence_length: int,
    depth: int,
    parallelism: Literal["row, column"],
    on_residual: bool,
    weight_type: UMUP_WEIGHT_TYPE,
):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(  # type: ignore[call-arg]
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            model_parallel_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            pipe_parallel_size=1,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    if parallelism == "column":
        linear_layer = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            umup_weight_type=weight_type,
            umup_on_residual=on_residual,
            topology=topology,
            parallel_output=False,
        )
    else:
        linear_layer = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            umup_weight_type=weight_type,
            umup_on_residual=on_residual,
            topology=topology,
            parallel_input=False,
            parallel_output=False,
        )

    linear_layer.umup_setup(effective_batch_size=batch_size * sequence_length, depth=depth)

    assert hasattr(linear_layer, "forward_multiplier")
    assert hasattr(linear_layer, "backward_multiplier")
    assert hasattr(linear_layer, "weight_grad_multiplier")
    assert hasattr(linear_layer, "bias_grad_multiplier")

    if weight_type == UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT:
        assert is_close(linear_layer.forward_multiplier, 1 / in_features)
        assert is_close(linear_layer.backward_multiplier, out_features ** (-0.5))
    else:
        assert is_close(linear_layer.forward_multiplier, in_features ** (-0.5))
        assert is_close(linear_layer.backward_multiplier, in_features ** (-0.5))

    assert is_close(linear_layer.weight_grad_multiplier, (sequence_length * batch_size) ** (-0.5))
    assert is_close(linear_layer.bias_grad_multiplier, (sequence_length * batch_size) ** (-0.5))

    assert hasattr(linear_layer.weight, "core_parameter_meta")
    assert isinstance(linear_layer.weight.core_parameter_meta, CoreParameterMeta)
    assert hasattr(linear_layer, "bias")
    assert hasattr(linear_layer.bias, "core_parameter_meta")
    assert isinstance(linear_layer.bias.core_parameter_meta, CoreParameterMeta)

    # check lr multipliers
    weight_lr_multiplier = 1.0
    bias_lr_multiplier = 1.0

    if weight_type == UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT:
        weight_lr_multiplier *= in_features ** (-0.5)

    if on_residual:
        weight_lr_multiplier *= depth ** (-0.5)
        bias_lr_multiplier *= depth ** (-0.5)

    assert is_close(linear_layer.weight.core_parameter_meta.lr_gain, weight_lr_multiplier)
    assert is_close(linear_layer.bias.core_parameter_meta.lr_gain, bias_lr_multiplier)

    # check unit scale for weight and bias
    assert has_scale(linear_layer.weight, 1.0)
    assert isinstance(linear_layer.bias, torch.Tensor)
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))

    x = torch.randn((batch_size, sequence_length, in_features), device=topology.device)
    y_grad = torch.randn((batch_size, sequence_length, out_features), device=topology.device)

    x.requires_grad = True
    y = linear_layer(x)
    y.backward(y_grad)

    assert x.grad is not None

    # check that scale for output, x_grad and w_grad are expected as of u-mup
    if weight_type == UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT:
        assert has_scale(y, in_features ** (-0.5)), y.std().item()
        assert has_scale(x.grad, 1.0), x.grad.std().item()
    else:
        assert has_scale(y, 1.0), y.std().item()
        assert has_scale(x.grad, (out_features / in_features) ** 0.5)

    assert linear_layer.weight.grad is not None
    assert has_scale(linear_layer.weight.grad, 1.0)
    assert has_scale(linear_layer.bias.grad, 1.0, rtol=0.069)  # type: ignore


@pytest.mark.parametrize("parallelism", ["column", "row"])
@pytest.mark.parametrize("on_residual", [False, True])
@pytest.mark.parametrize(
    "weight_type", [UMUP_WEIGHT_TYPE.INPUT_WEIGHT, UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT, UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT]
)
@pytest.mark.parametrize("model_parallel_size", [1, 2])
def test_umup_linear_layers(
    parallelism: Literal["row, column"], on_residual: bool, weight_type: UMUP_WEIGHT_TYPE, model_parallel_size: int
):
    in_features = 512
    out_features = 1024
    depth = 10
    batch_size = 16
    sequence_length = 64

    return_dict = dist_launcher(
        run_func=run_test_parallel_linear,
        world_size=model_parallel_size,
        model_parallel_size=model_parallel_size,
        master_port=find_free_port(),
        in_features=in_features,
        out_features=out_features,
        sequence_length=sequence_length,
        batch_size=batch_size,
        depth=depth,
        parallelism=parallelism,
        on_residual=on_residual,
        weight_type=weight_type,
    )

    assert return_dict is not None


def run_test_vocab_parallel_embedding(
    return_dict: dict,
    model_parallel_size: int,
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    sequence_length: int,
):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(  # type: ignore[call-arg]
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            model_parallel_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            pipe_parallel_size=1,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    embedding_layer = VocabParallelEmbedding(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim, finetunable_token_ids=[], topology=topology
    )

    embedding_layer.umup_setup(
        effective_batch_size=batch_size * sequence_length, depth=1
    )  # depth does not do anything for embedding layer

    assert hasattr(embedding_layer, "forward_multiplier")
    assert hasattr(embedding_layer, "weight_grad_multiplier")

    assert is_close(embedding_layer.forward_multiplier, 1.0)
    assert is_close(embedding_layer.weight_grad_multiplier, (batch_size * sequence_length / num_embeddings) ** -0.5)

    assert hasattr(embedding_layer.weight, "core_parameter_meta")
    assert isinstance(embedding_layer.weight.core_parameter_meta, CoreParameterMeta)

    assert is_close(embedding_layer.weight.core_parameter_meta.lr_gain, embedding_dim ** (-0.5))

    # check unit scale for weight
    assert has_scale(embedding_layer.weight, 1.0)

    x = torch.randint(0, num_embeddings, (batch_size, sequence_length), device=topology.device)
    y_grad = torch.randn((batch_size, sequence_length, embedding_dim), device=topology.device)

    y = embedding_layer(x)
    y.backward(y_grad)

    # check that scale for output, x_grad and w_grad are expected as of u-mup
    assert has_scale(y, 1.0)
    assert embedding_layer.weight.grad is not None
    assert has_scale(embedding_layer.weight.grad, 1.0)


@pytest.mark.parametrize("model_parallel_size", [1, 2])
def test_umup_embedding_layer(model_parallel_size: int):
    num_embeddings = 2048
    embedding_dim = 1024
    batch_size = 16
    sequence_length = 64

    return_dict = dist_launcher(
        run_func=run_test_vocab_parallel_embedding,
        world_size=model_parallel_size,
        model_parallel_size=model_parallel_size,
        master_port=find_free_port(),
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    assert return_dict is not None


@pytest.mark.parametrize("on_residual", [False, True])
@pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
def test_umup_norm(on_residual: bool, norm_type: Literal["layer_norm", "rms_norm"]):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch_size, seq_len = 2**5, 2**6

    config = LayerNormConfig(optimization_type=LayerNormOptimizationType.TORCH, layernorm_epsilon=1.0e-5)

    hidden_dim = 1024
    depth = 10
    batch_size = 16
    sequence_length = 64

    device = torch.device("cuda")

    norm: LayerNorm | RMSNorm

    if norm_type == "layer_norm":
        norm = LayerNorm(normalized_shape=hidden_dim, config=config, device=device, umup_on_residual=on_residual)
    else:
        norm = RMSNorm(dimensions=hidden_dim, config=config, device=device, umup_on_residual=on_residual)

    norm.umup_setup(effective_batch_size=batch_size * sequence_length, depth=10)

    assert hasattr(norm, "forward_multiplier")
    assert hasattr(norm, "backward_multiplier")
    assert hasattr(norm, "weight_grad_multiplier")

    assert is_close(norm.forward_multiplier, 1.0)
    assert is_close(norm.backward_multiplier, 1.0)
    assert is_close(norm.weight_grad_multiplier, (batch_size * sequence_length) ** (-0.5))

    assert hasattr(norm.weight, "core_parameter_meta")
    assert isinstance(norm.weight.core_parameter_meta, CoreParameterMeta)
    assert is_close(norm.weight.core_parameter_meta.lr_gain, depth ** (-0.5) if on_residual else 1.0)

    if norm_type == "layer_norm":
        assert hasattr(norm, "bias_grad_multiplier")
        assert is_close(norm.bias_grad_multiplier, (batch_size * sequence_length) ** (-0.5))

        assert hasattr(norm, "bias")
        assert hasattr(norm.weight, "core_parameter_meta")
        assert isinstance(norm.weight.core_parameter_meta, CoreParameterMeta)
        assert is_close(norm.bias.core_parameter_meta.lr_gain, depth ** (-0.5) if on_residual else 1.0)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x.requires_grad = True
    y = norm(x)
    y.backward(torch.randn_like(y))

    assert has_scale(y, 1.0)
    assert torch.allclose(norm.weight, torch.ones_like(norm.weight))
    assert x.grad is not None
    assert has_scale(x.grad, 1.0)
    assert norm.weight.grad is not None
    assert has_scale(norm.weight.grad, 1.0)

    if norm_type == "layer_norm":
        assert hasattr(norm, "bias")
        assert isinstance(norm.bias, torch.Tensor)
        assert norm.bias.grad is not None
        assert torch.allclose(norm.bias, torch.zeros_like(norm.bias))
        assert has_scale(norm.bias.grad, 1.0)


@pytest.mark.parametrize("umup_pre_norm", [True, False])
def test_umup_residual_split(umup_pre_norm: bool):
    umup_residual_mults = [1.0, 2.0, 3.0, 4.0]
    hidden_dim = 1024
    depth = 4
    batch_size = 16
    sequence_length = 64

    for layer_index in range(4):
        residual_split = NormedResidualSplit(
            umup_residual_mults=umup_residual_mults,
            umup_residual_layer_index=layer_index,
            umup_pre_norm=umup_pre_norm,
        )
        residual_split.umup_setup(depth=depth)

        assert hasattr(residual_split, "residual_scale")
        assert hasattr(residual_split, "skip_scale")

        x = torch.randn(batch_size, sequence_length, hidden_dim)
        x.requires_grad = True
        x_residual, x_skip = residual_split(x)
        grad_in = torch.randn_like(x)
        x_residual.backward(grad_in)
        x_skip.backward(grad_in)

        assert torch.allclose(x, x_residual)
        assert torch.allclose(x, x_skip)
        assert x.grad is not None
        assert has_scale(x.grad, residual_split.residual_scale + residual_split.skip_scale)


@pytest.mark.parametrize("umup_pre_norm", [True, False])
def test_umup_residual_add(umup_pre_norm: bool):
    umup_residual_mults = [1.0, 2.0, 3.0, 4.0]
    hidden_dim = 1024
    depth = 4
    batch_size = 16
    sequence_length = 64

    for layer_index in range(4):
        residual_add = NormedResidualAdd(
            umup_residual_mults=umup_residual_mults,
            umup_residual_layer_index=layer_index,
            umup_pre_norm=umup_pre_norm,
        )
        residual_add.umup_setup(depth=depth)

        assert hasattr(residual_add, "residual_scale")
        assert hasattr(residual_add, "skip_scale")

        x_residual = torch.randn(batch_size, sequence_length, hidden_dim)
        x_skip = torch.randn(batch_size, sequence_length, hidden_dim)
        x_residual.requires_grad = True
        x_skip.requires_grad = True
        y = residual_add(x_residual, x_skip)
        grad_in = torch.randn_like(y)
        y.backward(grad_in)

        assert has_scale(y, 1.0)
        assert x_residual.grad is not None
        assert x_skip.grad is not None
        assert torch.allclose(x_residual.grad, grad_in)
        assert torch.allclose(x_skip.grad, grad_in)


@pytest.mark.parametrize("mlp_type", ["standard", "swiglu"])
@pytest.mark.parametrize("mult", [1.0, 4.0])
@pytest.mark.parametrize("is_first_layer", [True, False])
@pytest.mark.parametrize("is_last_layer", [True, False])
@pytest.mark.parametrize("on_residual", [True, False])
def test_umup_mlp(
    mlp_type: Literal["standard", "swiglu"], mult: float, is_first_layer: bool, is_last_layer: bool, on_residual: bool
):
    hidden_dim = 512
    factor = 4.0
    batch_size = 16
    sequence_length = 64

    mlp: ParallelMLP | ParallelSwiGLUMLP

    if mlp_type == "standard":
        mlp = ParallelMLP(
            io_features=hidden_dim,
            intermediate_feature_factor=factor,
            bias=True,
            device=torch.device("cuda"),
            topology=None,
            umup_mult=mult,
            umup_is_first_layer=is_first_layer,
            umup_is_last_layer=is_last_layer,
            umup_on_residual=on_residual,
        )
    else:
        mlp = ParallelSwiGLUMLP(
            io_features=hidden_dim,
            intermediate_feature_factor=factor,
            bias=True,
            device=torch.device("cuda"),
            topology=None,
            umup_mult=mult,
            umup_is_first_layer=is_first_layer,
            umup_is_last_layer=is_last_layer,
            umup_on_residual=on_residual,
        )

    mlp.umup_setup()

    assert mlp.umup_mult == mult

    assert mlp.dense_in.umup_on_residual == on_residual
    if mlp_type == "swiglu":
        assert mlp.siglu_weight.umup_on_residual == on_residual
    assert mlp.dense_out.umup_on_residual == on_residual

    assert (
        mlp.dense_out.umup_weight_type == UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT
        if is_last_layer
        else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
    )
    assert (
        mlp.dense_in.umup_weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT
        if is_first_layer
        else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
    )
    if mlp_type == "swiglu":
        assert (
            mlp.siglu_weight.umup_weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT
            if is_first_layer
            else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
        )

    assert hasattr(mlp, "output_scale_factor")

    x = torch.randn(batch_size, sequence_length, hidden_dim).cuda()
    gate = torch.randn(batch_size, sequence_length, hidden_dim).cuda()
    if mlp_type == "swiglu":
        y = mlp._umup_siglu_forward(x, gate)
    else:
        y = mlp._umup_gelu_forward(x)
    assert has_scale(y, 1.0)
    # we don't assert backward properties here because we don't expect unit scale


# @pytest.mark.parametrize("kernel_type", ["torch", "flash_attention"])
@pytest.mark.parametrize("kernel_type", ["torch"])
@pytest.mark.parametrize("mult", [1.0, 4.0])
@pytest.mark.parametrize("is_first_layer", [True, False])
@pytest.mark.parametrize("is_last_layer", [True, False])
@pytest.mark.parametrize("on_residual", [True, False])
@pytest.mark.parametrize("qkv_in_one", [True, False])
def test_umup_attention(
    kernel_type: Literal["torch", "flash_attention"],
    mult: float,
    is_first_layer: bool,
    is_last_layer: bool,
    on_residual: bool,
    qkv_in_one: bool,
):
    hidden_dim = 1024
    batch_size = 16
    sequence_length = 512

    kernel = MaskedSoftmaxKernel.TORCH if kernel_type == "torch" else MaskedSoftmaxKernel.FLASH_ATTENTION

    attention = ParallelSelfAttention(
        hidden_size=hidden_dim,
        num_attention_heads=1,
        masked_softmax_config=MaskedSoftmaxConfig(kernel=kernel),
        causal=True,
        umup_is_first_layer=is_first_layer,
        umup_is_last_layer=is_last_layer,
        umup_on_residual=on_residual,
        rotary_config=RotaryConfig(dimensions=hidden_dim),
        umup_mult=mult,
        qkv_in_one=qkv_in_one,
    )

    assert attention.scaled_dot_product_attention.umup_mult == mult

    if qkv_in_one:
        assert attention.query_key_value.umup_on_residual == on_residual
        if is_first_layer:
            assert attention.query_key_value.umup_weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT
        else:
            assert attention.query_key_value.umup_weight_type == UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
    else:
        assert attention.query.umup_on_residual == on_residual
        assert attention.key.umup_on_residual == on_residual
        assert attention.value.umup_on_residual == on_residual
        if is_first_layer:
            assert attention.query.umup_weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT
            assert attention.key.umup_weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT
            assert attention.value.umup_weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT

    assert attention.dense.umup_on_residual == on_residual
    if is_last_layer:
        assert attention.dense.umup_weight_type == UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT
    else:
        assert attention.dense.umup_weight_type == UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT

    attention.scaled_dot_product_attention.umup_setup(avg_sequence_length=sequence_length)

    assert hasattr(attention.scaled_dot_product_attention, "output_scale_factor")

    query = torch.randn(sequence_length, batch_size, 1, hidden_dim).cuda()
    key = torch.randn(sequence_length, batch_size, 1, hidden_dim).cuda()
    value = torch.randn(sequence_length, batch_size, 1, hidden_dim).cuda()
    if kernel_type == "flash_attention":
        query, key, value = query.half(), key.half(), value.half()  # FA does not support FP32
    cum_seq_len = torch.tensor([0, sequence_length]).to(dtype=torch.int32, device=torch.device("cuda"))
    y = attention.scaled_dot_product_attention(query, key, value, cum_seq_len)
    assert has_scale(y, 1.0, 0.06), y.std()
    # we don't assert backward properties here because we don't expect unit scale
