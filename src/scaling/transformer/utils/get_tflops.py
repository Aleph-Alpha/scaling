from enum import Enum
from typing import Any

import torch.cuda

from scaling.core import Topology
from scaling.core.logging import logger
from scaling.core.topology.topology_config import ActivationCheckpointingType
from scaling.transformer.context.config import TransformerArchitectureConfig


def get_tflops_aleph_alpha(
    iter_time_s: float, topology: Any, transformer_architecture: TransformerArchitectureConfig
) -> float:
    """
    Computation is based on electra baseline with a few adjustments. Assumptions are:

        - From electra: An "operation" is a mathematical operation, not a machine instruction. So
        an "exp" takes one opp like and add, even though in practice an exp
        might be slower. This is not too bad an assumption because
        matrix-multiplies dominate the compute for most models, so minor details
        about activation functions don"t matter too much.
        - Adapted From electra: we count matrix-multiplies as (2*p -1)*m flops instead of p*m, as one might if
        considering fused multiply-add ops.
        This is approximated by (2 * p * m)

        For a matrix multiplication (n p) (p m) there are n×m  elements in the output matrix. Each of them is obtained
        by p multiplications (1 element from the first matrix and 1 from the second), then summing up. Since you have p
        products, you add p−1 of them to the first one.
        So the number of operations for one element in the output matrix is p multiplications and p−1
        additions, meaning 2p−1 FLOPS. Then for all elements, you get n×m×(2p−1) FLOPS.

        - an addition is a floating point operation (e.g. biases)

        - Backward pass takes the same number of FLOPs as forward pass. No exactly
        right (e.g., for softmax cross entropy loss the backward pass is faster).
        Importantly, it really is the same for matrix-multiplies,
        which is most of the compute costs anyway.
        - We assume "dense" embedding lookups (i.e., multiplication by a one-hot
        vector). On some hardware accelerators, these dense operations are
        actually faster than sparse lookups.

    """
    # GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
    gelu_flops = 8

    # min/subtract (for stability), exp, sum, divide
    softmax_flops = 5

    # random number, >=, multiply activations by dropout mask, multiply activations by
    # correction (1 / (1 - dropout_rate))
    dropout_flops = 4

    # compute mean activation (sum), compute variance of activation
    # (square and sum), bias (add), scale (multiply)
    layer_norm_flops = 5

    # (query * cos_q) + (rotate_half(query) * sin_q), (key * cos_k) + (rotate_half(key) * sin_k)
    rotary_flops = 6

    layer_flops_per_token = dict(
        attention_input_layer_norm=layer_norm_flops,
        attention_kqv_matrix_multiply=3
        * 2
        * transformer_architecture.hidden_size
        * transformer_architecture.hidden_size,
        kqv_bias=3 * transformer_architecture.hidden_size,
        attention_scores=2 * transformer_architecture.hidden_size * transformer_architecture.sequence_length,
        attention_softmax=softmax_flops
        * transformer_architecture.sequence_length
        * transformer_architecture.num_attention_heads,
        attention_rotary=rotary_flops * transformer_architecture.hidden_size,
        attention_probs_dropout=dropout_flops
        * transformer_architecture.sequence_length
        * transformer_architecture.num_attention_heads,
        attention_scale=transformer_architecture.sequence_length * transformer_architecture.num_attention_heads,
        attention_weighted_avg_values=2
        * transformer_architecture.sequence_length
        * transformer_architecture.hidden_size,
        attention_dense_matrix_multiply=2 * transformer_architecture.hidden_size * transformer_architecture.hidden_size,
        attention_dense_bias=transformer_architecture.hidden_size,
        attention_dropout=dropout_flops * transformer_architecture.hidden_size,
        attention_residual=transformer_architecture.hidden_size,
        mlp_post_attention_layer_norm=layer_norm_flops,
        mlp_dense_in_matrix_multiply=2
        * transformer_architecture.hidden_size
        * transformer_architecture.hidden_size
        * 4,
        mlp_dense_in_bias=transformer_architecture.hidden_size * 4,
        mlp_act=gelu_flops * transformer_architecture.hidden_size * 4,
        mlp_dense_output_matrix_multiply=2
        * transformer_architecture.hidden_size
        * transformer_architecture.hidden_size
        * 4,
        mlp_dense_output_bias=transformer_architecture.hidden_size,
        mlp_dropout=dropout_flops * transformer_architecture.hidden_size,
        mlp_residual=transformer_architecture.hidden_size,
    )
    layer_flops_total = (
        transformer_architecture.num_layers
        * topology.config.global_batch_size
        * transformer_architecture.sequence_length
        * sum(layer_flops_per_token.values())
    )
    final_layer_norm_flops_total = layer_norm_flops
    lm_head_flops = (
        2
        * topology.config.global_batch_size
        * transformer_architecture.sequence_length
        * transformer_architecture.hidden_size
        * transformer_architecture.vocab_size
    )

    embedding_flops_total = 0

    train_multiply_for_backward = 3

    assert topology.config.world_size is not None
    tflops = (
        train_multiply_for_backward
        * (embedding_flops_total + layer_flops_total + final_layer_norm_flops_total + lm_head_flops)
        / (iter_time_s * topology.config.world_size * (10.0**12))
    )

    return tflops


def get_tflops_electra(
    iter_time_s: float, topology: Any, transformer_architecture: TransformerArchitectureConfig
) -> float:
    """
    Computation based on https://github.com/google-research/electra/blob/master/flops_computation.py#L38

    "
    We checked this code with TensorFlow"s FLOPs counting, although we had to
    correct for this issue: https://github.com/tensorflow/tensorflow/issues/22071
    Assumptions going into the FLOPs counting
        - An "operation" is a mathematical operation, not a machine instruction. So
        an "exp" takes one opp like and add, even though in practice an exp
        might be slower. This is not too bad an assumption because
        matrix-multiplies dominate the compute for most models, so minor details
        about activation functions don"t matter too much. Similarly, we count
        matrix-multiplies as 2*m*n flops instead of m*n, as one might if
        considering fused multiply-add ops.
        - Backward pass takes the same number of FLOPs as forward pass. No exactly
        right (e.g., for softmax cross entropy loss the backward pass is faster).
        Importantly, it really is the same for matrix-multiplies, which is most of
        the compute costs anyway.
        - We assume "dense" embedding lookups (i.e., multiplication by a one-hot
        vector). On some hardware accelerators, these dense operations are
        actually faster than sparse lookups.
    "

    """
    # GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
    gelu_flops = 8

    # max/subtract (for stability), exp, sum, divide
    softmax_flops = 5

    # random number, >=, multiply activations by dropout mask, multiply activations by
    # correction (1 / (1 - dropout_rate))
    dropout_flops = 4

    # compute mean activation (sum), compute variance of activation
    # (square and sum), bias (add), scale (multiply)
    layer_norm_flops = 5

    # (query * cos_q) + (rotate_half(query) * sin_q), (key * cos_k) + (rotate_half(key) * sin_k)
    rotary_flops = 4

    layer_flops_per_token = dict(
        attention_input_layer_norm=layer_norm_flops,
        attention_kqv_matrix_multiply=3
        * 2
        * transformer_architecture.hidden_size
        * transformer_architecture.hidden_size,
        kqv_bias=3 * transformer_architecture.hidden_size,
        attention_scores=2 * transformer_architecture.hidden_size * transformer_architecture.sequence_length,
        attention_softmax=softmax_flops
        * transformer_architecture.sequence_length
        * transformer_architecture.num_attention_heads,
        attention_rotary=rotary_flops * transformer_architecture.hidden_size,
        attention_probs_dropout=dropout_flops
        * transformer_architecture.sequence_length
        * transformer_architecture.num_attention_heads,
        attention_scale=transformer_architecture.sequence_length * transformer_architecture.num_attention_heads,
        attention_weighted_avg_values=2
        * transformer_architecture.sequence_length
        * transformer_architecture.hidden_size,
        attention_dense_matrix_multiply=2 * transformer_architecture.hidden_size * transformer_architecture.hidden_size,
        attention_dense_bias=transformer_architecture.hidden_size,
        attention_dropout=dropout_flops * transformer_architecture.hidden_size,
        attention_residual=transformer_architecture.hidden_size,
        mlp_post_attention_layer_norm=layer_norm_flops,
        mlp_dense_in_matrix_multiply=2
        * transformer_architecture.hidden_size
        * transformer_architecture.hidden_size
        * 4,
        mlp_dense_in_bias=transformer_architecture.hidden_size * 4,
        mlp_act=gelu_flops * transformer_architecture.hidden_size * 4,
        mlp_dense_output_matrix_multiply=2
        * transformer_architecture.hidden_size
        * transformer_architecture.hidden_size
        * 4,
        mlp_dense_output_bias=transformer_architecture.hidden_size,
        mlp_dropout=dropout_flops * transformer_architecture.hidden_size,
        mlp_residual=transformer_architecture.hidden_size,
    )
    layer_flops_total = (
        transformer_architecture.num_layers
        * topology.config.global_batch_size
        * transformer_architecture.sequence_length
        * sum(layer_flops_per_token.values())
    )
    final_layer_norm_flops_total = layer_norm_flops
    lm_head_flops = (
        2
        * topology.config.global_batch_size
        * transformer_architecture.sequence_length
        * transformer_architecture.hidden_size
        * transformer_architecture.vocab_size
    )

    embedding_flops_total = (
        2
        * topology.config.global_batch_size
        * transformer_architecture.sequence_length
        * transformer_architecture.hidden_size
        * transformer_architecture.vocab_size
    )

    train_multiply_for_backward = 2

    assert topology.config.world_size is not None
    tflops = (
        train_multiply_for_backward
        * (embedding_flops_total + layer_flops_total + final_layer_norm_flops_total + lm_head_flops)
        / (iter_time_s * topology.config.world_size * (10.0**12))
    )

    return tflops


def get_tflops_bloom(
    iter_time_s: float, topology: Any, transformer_architecture: TransformerArchitectureConfig
) -> float:
    """
    Based on APPENDIX: FLOATING-POINT OPERATIONS of
    https://arxiv.org/pdf/2104.04473.pdf

    In this section, we describe how we calculate the number of floating point operations (FLOPs) in a model.
    We consider a language model with 𝑙 transformer layers, hidden size ℎ, sequence length 𝑠, vocabulary size 𝑉 ,
    and training batch size 𝐵. A 𝐴𝑚×𝑘 ×𝑋𝑘×𝑛 matrix multiplication requires 2𝑚 ×𝑘 ×𝑛 FLOPs
    (factor of 2 needed to account for multiplies and adds).
    A transformer layer consists of an attention block followed by
    a 2-layer feed-forward network. For the attention block, the main
    FLOP contributors are the key, query, and value transformation
    (6𝐵𝑠ℎ² operations), attention matrix computation (2𝐵𝑠²ℎ operations), attention over values (2𝐵𝑠²ℎ operations),
    and post-attention linear projection (2𝐵𝑠ℎ² operations). The feed-forward network
    increases the hidden size to 4ℎ and then reduces it back to ℎ; this
    requires 16𝐵𝑠ℎ2 FLOPs. Summing these together, each transformer
    layer results in 24𝐵𝑠ℎ² + 4𝐵𝑠²ℎ FLOPs for the forward pass. The
    backward pass requires double the number of FLOPs since we
    need to calculate the gradients with respect to both input and
    weight tensors. In addition, we are using activation recomputation,
    which requires an additional forward pass before the backward
    pass. As a result, the total number of FLOPs per transformer layer
    is 4 × (24𝐵𝑠ℎ² + 4𝐵𝑠²ℎ) = 96𝐵𝑠ℎ² (1 + 𝑠 / 6ℎ).
    The other main contributor to the FLOP count is the logit layer in
    the language model head, which transforms features of dimension
    ℎ to the vocabulary dimension 𝑉 . The required FLOPs for this
    operation is 2𝐵𝑠ℎ𝑉 in the forward pass and 4𝐵𝑠ℎ𝑉 in the backward
    pass, resulting in 6𝐵𝑠ℎ𝑉 FLOPs in total.
    Thus, for a transformer model with 𝑙 transformer layers, the
    total number of floating-point operations is:
    96𝐵𝑠𝑙ℎ2 (1 + 𝑠 / 6ℎ + 𝑉 / 16𝑙ℎ).
    """

    # In addition, we are using activation recomputation,
    # which requires an additional forward pass before the backward
    # pass. As a result, the total number of FLOPs per transformer layer
    checkpoint_activations_factor = (
        3 if topology.config.activation_checkpointing_type == ActivationCheckpointingType.DISABLED else 4
    )

    assert topology.config.world_size is not None
    tflops = (
        (
            24  # KQV (6𝐵𝑠ℎ² operations), post-attention linear projection (2𝐵𝑠ℎ² operations),
            # MLP (16𝐵𝑠ℎ² operations), assuming a multiplier to 4 h
            * checkpoint_activations_factor
            * topology.config.global_batch_size
            * transformer_architecture.sequence_length
            * transformer_architecture.num_layers
            * (transformer_architecture.hidden_size**2)
        )
        + (
            4  # Attention matrix (2𝐵𝑠²ℎ operations), Attention values (2𝐵𝑠²ℎ operations)
            * checkpoint_activations_factor
            * topology.config.global_batch_size
            * (transformer_architecture.sequence_length**2)
            * transformer_architecture.hidden_size
            * transformer_architecture.num_layers
        )
        + (
            # LM HEAD (6𝐵𝑠ℎV operations)
            (2 + checkpoint_activations_factor)
            * topology.config.global_batch_size
            * transformer_architecture.sequence_length
            * transformer_architecture.hidden_size
            * transformer_architecture.vocab_size
        )
    ) / (iter_time_s * topology.config.world_size * (10.0**12))

    return tflops


def get_tflops_megatron(
    parameter_count: int, iter_time_s: float, topology: Any, transformer_architecture: TransformerArchitectureConfig
) -> float:
    assert topology.config.world_size is not None
    ff = topology.config.global_batch_size * transformer_architecture.sequence_length * parameter_count * 2 * 3
    attn = (
        topology.config.global_batch_size
        * transformer_architecture.sequence_length
        * transformer_architecture.sequence_length
        * transformer_architecture.hidden_size
        * transformer_architecture.num_layers
        * 60
    )
    tflops = (ff + attn) / (iter_time_s * topology.config.world_size * (10.0**12))

    return tflops


class HardwareType(Enum):
    A100 = "a100"
    H100 = "h100"
    RTX3090 = "rtx3090"
    RTX4090 = "rtx4090"
    DEFAULT = "default"
    # TODO add more

    @property
    def max_tflops(self) -> float:
        """
        Mappings for Maximum Throughput numbers of each GPU.
        Only includes FP16 for now.
        """
        max_tflop_mapping = {"a100": 312e12, "h100": 989.4e12, "rtx3090": 35.58e12, "rtx4090": 82.58, "default": 0.0}
        return max_tflop_mapping[self.value]

    @classmethod
    def get_via_torch(cls) -> "HardwareType":
        cuda_device_name = torch.cuda.get_device_name()
        cuda_device_name_clean = cuda_device_name.replace(" ", "").lower().strip().replace("nvidia", "")
        result = next((x for x in cls if x != cls.DEFAULT and x.value.lower() in cuda_device_name_clean), None)
        if result is None:
            logger.warning(f"cuda device {cuda_device_name} does not match any known HardwareType")
            return cls.DEFAULT
        return result


def get_model_flop_utilization_palm(
    iter_time_s: float,
    parameter_count: int,
    topology: Topology,
    transformer_architecture: TransformerArchitectureConfig,
) -> float:
    """
    Computation based on PaLM: https://arxiv.org/abs/2204.02311

    In PaLM they recognize that measuring the training efficiency via the hardware FLOPs utilization (HFU) is a poor
    metric of the actual training efficiency of a model because of its system- and implementation-dependence and design
    choices in the compiler which can result in different number of operations. They propose a more meaningful metric
    called Model FLOPs Utilization (MFU). It consists of the ratio of the observed throughput (tokens-per-second)
    relative to the theoretical maximum throughput of a system operating at peak FLOPs. While the “theoretical maximum”
    throughput only accounts for the required operations to compute the forward and backward passes, and not
    rematerialization.

    You take the theoretical peak matmul throughput of P FLOPs per second (e.g. A100 GPUs with 312 peak matmul TFLOP/s)
    Then the model FLOPs utilization is the ratio of the achieved throughput in tokens per second to the
    theoretical peak throughput R = P / (6N + 12LHQT)

    MFU = Actual Tokens-per-second / R
    """
    hardware = HardwareType.get_via_torch()
    assert topology.config.global_batch_size is not None
    tokens_per_second = (topology.config.global_batch_size * transformer_architecture.sequence_length) / iter_time_s
    theoretical_peak_matmul_throughput = hardware.max_tflops * topology.config.world_size
    attention_flops = (
        12
        * transformer_architecture.num_layers
        * transformer_architecture.hidden_size
        * transformer_architecture.sequence_length
    )
    model_flops = 6 * parameter_count + attention_flops
    theoretical_peak_throughput = theoretical_peak_matmul_throughput / model_flops
    model_flops_utilization = tokens_per_second / theoretical_peak_throughput
    return model_flops_utilization
