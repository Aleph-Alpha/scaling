import math

import torch

from .parameter_meta import (
    UMUP_WEIGHT_TYPE,
    CoreParameterMeta,
    UMuPParameterMeta,
)


class UMuParametrization:
    @staticmethod
    def get_parameter_multipliers(
        parameter_meta: CoreParameterMeta,
        model_parallel_size: int,
        effective_batch_size: int,
        depth: int,
    ) -> tuple[float, float, float, float]:
        """returns scaling factors for forward_multiplier, backward_multiplier, grad_multiplier
        and lr based on parameter info
        """

        assert isinstance(parameter_meta.umup_meta, UMuPParameterMeta)
        umup_meta = parameter_meta.umup_meta
        local_shape = parameter_meta.local_shape
        is_model_parallel = parameter_meta.is_model_parallel
        model_parallel_dimension = parameter_meta.model_parallel_dimension

        # bias, norm weight
        if umup_meta.weight_type in (UMUP_WEIGHT_TYPE.NORM, UMUP_WEIGHT_TYPE.BIAS):
            fwd_factor, bwd_factor, grad_factor, lr_factor = 1.0, 1.0, effective_batch_size**-0.5, 1.0
        # embedding
        elif umup_meta.weight_type == UMUP_WEIGHT_TYPE.INPUT_EMBEDDING:
            hidden_size = local_shape[1]
            if is_model_parallel:
                vocab_size = local_shape[0] * model_parallel_size
            fwd_factor, bwd_factor, grad_factor, lr_factor = (
                1.0,
                1.0,
                (effective_batch_size / vocab_size) ** -0.5,
                hidden_size**-0.5,
            )
        # linear layer weights
        else:
            if is_model_parallel and model_parallel_dimension == 1:
                n_in = local_shape[1] * model_parallel_size
                n_out = local_shape[0]
            elif is_model_parallel and model_parallel_dimension == 0:
                n_in = local_shape[1]
                n_out = local_shape[0] * model_parallel_size
            else:
                n_in = local_shape[1]
                n_out = local_shape[0]

            if umup_meta.weight_type == UMUP_WEIGHT_TYPE.INPUT_WEIGHT:
                fwd_factor, bwd_factor, grad_factor, lr_factor = (
                    1 / math.sqrt(n_in),
                    1 / math.sqrt(n_in),
                    effective_batch_size**-0.5,
                    1.0,
                )
            elif umup_meta.weight_type == UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT:
                fwd_factor, bwd_factor, grad_factor, lr_factor = (
                    1 / math.sqrt(n_in),
                    1 / math.sqrt(n_in),
                    effective_batch_size**-0.5,
                    1.0 / math.sqrt(n_in),
                )
            elif umup_meta.weight_type == UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT:
                fwd_factor, bwd_factor, grad_factor, lr_factor = (
                    1 / n_in,
                    1 / math.sqrt(n_out),
                    effective_batch_size**-0.5,
                    1.0,
                )
            else:
                raise ValueError
        if umup_meta.on_residual:
            depth_scale = UMuParametrization.get_depth_scale(depth)
            lr_factor *= depth_scale
        return fwd_factor, bwd_factor, grad_factor, lr_factor

    @staticmethod
    def apply_umup_to_weight(
        weight: torch.Tensor,
        model_parallel_size: int,
        effective_batch_size: int,
        depth: int,
        reinitialize: bool = True,
    ) -> None:
        assert hasattr(weight, "core_parameter_meta")
        assert isinstance(weight.core_parameter_meta, CoreParameterMeta)
        umup_meta = weight.core_parameter_meta.umup_meta
        assert isinstance(umup_meta, UMuPParameterMeta)

        (
            forward_multiplier_scaling_factor,
            backward_multiplier_scaling_factor,
            grad_multiplier_scaling_factor,
            lr_scaling_factor,
        ) = UMuParametrization.get_parameter_multipliers(
            parameter_meta=weight.core_parameter_meta,
            model_parallel_size=model_parallel_size,
            effective_batch_size=effective_batch_size,
            depth=depth,
        )

        # set final multiplier values for the parameter
        umup_meta.forward_multiplier = forward_multiplier_scaling_factor
        umup_meta.backward_multiplier = backward_multiplier_scaling_factor
        umup_meta.grad_multiplier = grad_multiplier_scaling_factor
        umup_meta.lr_multiplier = lr_scaling_factor

        # set correct learning rate gain on parameter meta
        weight.core_parameter_meta.scale_lr_gain(umup_meta.lr_multiplier)

        # re-initialize weight
        if reinitialize:
            if umup_meta.weight_type == UMUP_WEIGHT_TYPE.NORM:
                torch.nn.init.constant_(weight, 1.0)
            elif umup_meta.weight_type == UMUP_WEIGHT_TYPE.BIAS:
                torch.nn.init.constant_(weight, 0.0)
            else:
                torch.nn.init.normal_(weight)

    @staticmethod
    def get_umup_residual_scales(
        residual_mults: list[float], residual_layer_index: int, pre_norm: bool, depth: int
    ) -> tuple[float, float]:
        """
        Takes global list of residual multipliers and returns residual and skip scale based on u-mup
        """
        depth_scale = UMuParametrization.get_depth_scale(depth)
        scaled_residual_mults = [r * depth_scale for r in residual_mults]
        scaled_residual_variances = [r**2 for r in scaled_residual_mults]
        residual_sum_var = 1 + sum(scaled_residual_variances[:residual_layer_index])
        residual_current_var = scaled_residual_variances[residual_layer_index]

        if pre_norm:
            tau = residual_current_var / residual_sum_var
        else:
            tau = residual_current_var

        residual_scale = (tau / (tau + 1)) ** 0.5
        skip_scale = (1 / (tau + 1)) ** 0.5

        return residual_scale, skip_scale

    @staticmethod
    def get_umup_transformer_residual_mults(
        residual_mult: float,
        residual_attn_ratio: float,
        num_layers: int,
    ) -> list[float]:
        """
        returns list of residual multipliers for a transformer given residual mult and attn ratio
        """

        mlp_residual_mult = residual_mult * (2 / (1 + residual_attn_ratio**2)) ** 0.5

        attention_residual_mult = residual_attn_ratio * mlp_residual_mult

        transformer_block_residual_mults = [attention_residual_mult, mlp_residual_mult]

        transformer_residual_mults = transformer_block_residual_mults * num_layers

        return transformer_residual_mults

    @staticmethod
    def get_umup_gelu_scales(gelu_mult: float) -> float:
        output_scale = _logarithmic_interpolation(
            alpha=1 / (1 + 0.25 / gelu_mult**2),  # = sigmoid(log(4 * mult**2))
            lower=2**1,
            upper=(2 / (1 - 1 / math.pi)) ** 0.5,
        )
        return output_scale

    @staticmethod
    def get_umup_swiglu_scales(swiglu_mult: float) -> float:
        output_scale = 1 / _logarithmic_interpolation(
            alpha=1 / (1 + 1 / swiglu_mult**2),  # = sigmoid(log(mult**2))
            lower=2**-1,
            upper=2**-0.5,
        )
        return output_scale

    # TODO: add non-causal option
    @staticmethod
    def get_umup_attention_scales(
        head_dim: int,
        sequence_length: int,
        mult: float,
        dropout_p: float,
    ) -> tuple[float, float]:
        query_key_scale_factor = mult / head_dim
        output_scale_factor = (1 - dropout_p) ** 0.5
        # empirical model of attention output std given mult and seq_len (see notebook)
        output_scale_factor /= _logarithmic_interpolation(
            alpha=1 / (1 + 4 * head_dim / mult**2),  # = sigmoid(log(mult**2 / (4 * head_dim)))
            lower=math.sqrt(math.log(sequence_length) / sequence_length),
            upper=1,
        )
        return query_key_scale_factor, output_scale_factor

    @staticmethod
    def get_umup_cross_entropy_scales(mult: float, vocab_size: int) -> tuple[float, float]:
        return mult, vocab_size / (vocab_size - 1) ** 0.5

    @staticmethod
    def get_depth_scale(depth: int) -> float:
        return depth**-0.5


def _logarithmic_interpolation(alpha: float, lower: float, upper: float) -> float:
    return math.exp(alpha * math.log(upper) + (1 - alpha) * math.log(lower))
