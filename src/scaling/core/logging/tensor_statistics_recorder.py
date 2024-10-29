import enum
import fnmatch
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Callable, DefaultDict, Literal, Optional

import numpy as np
import torch
from pydantic import Field

from scaling.core.config import BaseConfig
from scaling.core.context import BaseContext
from scaling.core.logging.logging import _LoggerType
from scaling.core.logging.logging import logger as default_logger
from scaling.core.nn.parallel_module import ParallelModule
from scaling.core.nn.parallel_module.inference_module import InferenceModule

E4M3_SMALLEST = torch.tensor(2**-9)
E5M2_SMALLEST = torch.tensor(2**-16)


class TensorStatistics(enum.Enum):
    # Standard statistics
    L1 = "l1"
    MEAN = "mean"
    STD = "std"
    SKEW = "skew"
    KURTOSIS = "kurtosis"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    ABS_STD = "abs_std"
    ABS_MAX = "abs_max"
    ABS_MIN = "abs_min"
    ABS_MEDIAN = "abs_median"
    # Entropy statistic for softmax output, can be used to detect attention entropy collapse
    # Section 3.1.1 https://arxiv.org/pdf/2309.14322
    ENTROPY = "entropy"
    # Underflow/Overflow/Nan/Inf statistics
    PCT_ZERO = "pct_zero"
    PCT_UNDERFLOW_E4M3 = "pct_underflow_e4m3"
    PCT_UNDERFLOW_E5M2 = "pct_underflow_e5m2"
    PCT_POSINF = "pct_posinf"
    PCT_NEGINF = "pct_neginf"
    PCT_NAN = "pct_nan"
    # Outlier metrics following
    # https://blog.allenai.org/investigating-pretraining-dynamics-and-stability-with-olmo-checkpoints-ece6f0c4947a
    MAX_SPARSITY = "max_sparsity"
    NORM_SPARSITY = "norm_sparsity"
    # Quantile statistics
    QUANTILE_1 = "quantile_1"
    QUANTILE_10 = "quantile_10"
    QUANTILE_25 = "quantile_25"
    QUANTILE_75 = "quantile_75"
    QUANTILE_90 = "quantile_90"
    QUANTILE_99 = "quantile_99"
    ABS_QUANTILE_1 = "abs_quantile_1"
    ABS_QUANTILE_10 = "abs_quantile_10"
    ABS_QUANTILE_25 = "abs_quantile_25"
    ABS_QUANTILE_75 = "abs_quantile_75"
    ABS_QUANTILE_90 = "abs_quantile_90"
    ABS_QUANTILE_99 = "abs_quantile_99"

    default_statistics = [
        MEAN,
        STD,
        L1,
        ABS_STD,
        MAX,
        MIN,
        MEDIAN,
        ABS_MAX,
        ABS_MIN,
        ABS_MEDIAN,
    ]


TENSOR_STATISTICS_FUNCTIONS = {
    TensorStatistics.MEAN: lambda x: x.mean().item(),
    TensorStatistics.STD: lambda x: x.std().item(),
    TensorStatistics.SKEW: lambda x: torch.mean(torch.pow(((x - x.mean()) / x.std()).float(), 3.0)).item(),
    TensorStatistics.KURTOSIS: lambda x: (
        torch.mean(torch.pow(((x - x.mean()) / x.std(correction=0)).float(), 4.0)) - 3.0
    ).item(),
    TensorStatistics.L1: lambda x: (x.norm(p=1) / x.numel()).item(),
    TensorStatistics.ABS_STD: lambda x: x.abs().std().item(),
    TensorStatistics.MAX: lambda x: x.max().item(),
    TensorStatistics.MIN: lambda x: x.min().item(),
    TensorStatistics.MEDIAN: lambda x: x.float().median().item(),
    TensorStatistics.ABS_MAX: lambda x: x.abs().max().item(),
    TensorStatistics.ABS_MIN: lambda x: x.abs().min().item(),
    TensorStatistics.ABS_MEDIAN: lambda x: x.abs().float().median().item(),
    TensorStatistics.ENTROPY: lambda x: -torch.sum(x * torch.log(x + 1e-9), dim=-1).mean().item(),
    TensorStatistics.PCT_ZERO: lambda x: 1.0 - (x.count_nonzero() / x.numel()).item(),
    TensorStatistics.PCT_UNDERFLOW_E4M3: lambda x: (torch.count_nonzero(x.abs() <= E4M3_SMALLEST) / x.numel()).item(),
    TensorStatistics.PCT_UNDERFLOW_E5M2: lambda x: (torch.count_nonzero(x.abs() <= E5M2_SMALLEST) / x.numel()).item(),
    TensorStatistics.PCT_POSINF: lambda x: (x.isposinf().view(-1).sum() / x.numel()).item(),
    TensorStatistics.PCT_NEGINF: lambda x: (x.isneginf().view(-1).sum() / x.numel()).item(),
    TensorStatistics.PCT_NAN: lambda x: (x.isnan().view(-1).sum() / x.numel()).item(),
    # see https://blog.allenai.org/investigating-pretraining-dynamics-and-stability-with-olmo-checkpoints-ece6f0c4947a
    TensorStatistics.MAX_SPARSITY: lambda x: (lambda x, median: (x.abs().max() - median) / median)(
        x, x.abs().float().median()
    ).item(),
    TensorStatistics.NORM_SPARSITY: lambda x: (np.sqrt(x.numel()) * x.norm(p=2) / x.norm(p=1)).item(),
    # TODO: all operations below involve sorting the tensor once -> inefficient :/
    TensorStatistics.QUANTILE_1: lambda x: x.float().quantile(0.01).item(),
    TensorStatistics.QUANTILE_10: lambda x: x.float().quantile(0.10).item(),
    TensorStatistics.QUANTILE_25: lambda x: x.float().quantile(0.25).item(),
    TensorStatistics.QUANTILE_75: lambda x: x.float().quantile(0.75).item(),
    TensorStatistics.QUANTILE_90: lambda x: x.float().quantile(0.90).item(),
    TensorStatistics.QUANTILE_99: lambda x: x.float().quantile(0.99).item(),
    TensorStatistics.ABS_QUANTILE_1: lambda x: x.abs().float().quantile(0.01).item(),
    TensorStatistics.ABS_QUANTILE_10: lambda x: x.abs().float().quantile(0.10).item(),
    TensorStatistics.ABS_QUANTILE_25: lambda x: x.abs().float().quantile(0.25).item(),
    TensorStatistics.ABS_QUANTILE_75: lambda x: x.abs().float().quantile(0.75).item(),
    TensorStatistics.ABS_QUANTILE_90: lambda x: x.abs().float().quantile(0.90).item(),
    TensorStatistics.ABS_QUANTILE_99: lambda x: x.abs().float().quantile(0.99).item(),
}


def statistics_fcts_for_name_pattern(
    statistics_config: list[tuple[str, list["TensorStatistics"]]], name_pattern: str
) -> Optional[dict[str, Callable]]:
    statistics_ftcs = None
    for pattern, stats in statistics_config:
        if fnmatch.fnmatch(name_pattern, pattern):
            statistics_ftcs = {stat.value: TENSOR_STATISTICS_FUNCTIONS[stat] for stat in stats}
            break
    return statistics_ftcs


class TensorStatisticsRecorderConfig(BaseConfig):
    interval: int = Field(
        100,
        description="trace metrics every interval steps",
    )

    statistics: list[tuple[str, list[TensorStatistics]]] = Field(
        default_factory=lambda: [("*", TensorStatistics.default_statistics)],
        description="List of globbing patterns for module names to statistics to track for matching modules. "
        "First match wins.",
    )

    include_module_type: bool = Field(
        False,
        description="Include the module type in the name for activation statistics tracking.",
    )


class _TensorTracker:
    """
    A singleton class to track tensors and register hooks for monitoring tensor statistics.

    This class provides methods to trace tensors and register hooks that can be used to monitor
    tensor statistics during forward and backward passes in pytorch.

    Attributes:
        _instance (_TensorTracker): The singleton instance of the class.
        hooks (list): A list of hook functions to be called during the forward pass.
        backward_hooks (list): A list of hook functions to be called during the backward pass.
        _name_counter (defaultdict): A counter to keep track of the number of times each tensor name has been used.

    Methods:
        __new__(cls):
            Creates/reuses and returns the singleton instance of the class.

        trace_tensor(tensor: torch.Tensor, name: str):
            Traces a tensor by invoking all hooks with the tensor and its name and registering a backward hook if the
            tensor requires gradient.

        register_hook(hook_fn):
            Context manager that registers a forward hook function and ensures the
            forward/backward hooks are removed after use.
    """

    # Singleton class to track tensors
    _instance = None

    def __new__(cls) -> "_TensorTracker":
        if cls._instance is None:
            cls._instance = super(_TensorTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "__initialized"):
            self.hooks: list = []
            self.backward_hooks: list = []
            self.name_counter: DefaultDict[str, int] = defaultdict(int)
            self.__initialized = True

    def trace_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """
        Registers a tensor for tracking by invoking all hooks with the tensor and its name.

        This method iterates over all hooks stored in `self.hooks` and calls each hook with the provided
        tensor and its associated name. If the tensor requires gradient, a backward hook is registered.

        Args:
            tensor (torch.Tensor): The tensor to be tracked.
            name (str): The name associated with the tensor.
        """
        for hook in self.hooks:
            name = f"tensor_statistics/counter.{self.name_counter[name]}.{name}"
            hook(tensor=tensor, name=name, mode="forward")
            if tensor.requires_grad:

                def backward_hook_fct(grad: torch.Tensor) -> None:
                    hook(tensor=grad, name=name, mode="backward")

                backward_hook = tensor.register_hook(backward_hook_fct)
                self.backward_hooks.append(backward_hook)
        self.name_counter[name] += 1

    @contextmanager
    def register_hook(self, hook_fn: Callable) -> Iterator[None]:
        """
        Registers a hook function to all tracked tensors and ensures the hooks are removed after use.

        Args:
            hook_fn (callable): The hook function to be registered. It should accept the tensor name and
                                mode as keyword arguments.

        Yields:
            None: This function is a context manager and yields control back to the caller.
        """
        try:
            self.hooks.append(hook_fn)
            yield
        finally:
            self.hooks = []
            for backward_hook in self.backward_hooks:
                backward_hook.remove()
            self.name_counter = defaultdict(int)


tensor_tracker = _TensorTracker()


class TensorStatisticsRecorder:
    """
    A class to record tensor statistics for a given model during training and inference.

    Attributes:
        config (TensorStatisticsRecorderConfig): Configuration for tensor statistics recording.
        context (BaseContext): Context of the training or inference process.
        model (ParallelModule | InferenceModule): The model whose tensor statistics are to be recorded.
        module_to_name_dict (dict): A dictionary mapping modules to their names.

    Methods:
        register_parameter_hooks():
            Registers hooks to record statistics for model parameters during forward and backward passes.

        trace():
            Context manager to trace the model's forward and backward passes, recording tensor statistics.

        _get_statistics_fcts(name: str):
            Retrieves the statistics functions to be applied based on the parameter or module name.

        activation_hook(
            module: torch.nn.Module,
            input: torch.Tensor,
            output: Union[torch.Tensor, tuple],
            mode: Literal["forward", "backward"] = "forward"
        ):
            Hook function to record activation statistics during forward and backward passes.

        tensor_hook(tensor: torch.Tensor, name: str, mode: Literal["forward, backward"]):
            Hook function to record parameter/tensor statistics during forward and backward passes.
    """

    def __init__(
        self,
        config: TensorStatisticsRecorderConfig,
        context: BaseContext,
        model: ParallelModule | InferenceModule,
        logger: _LoggerType | None = None,
    ):
        self.config = config
        self.context = context
        self.model = model
        self.logger = logger if logger else default_logger

        # A dictionary mapping module to its name (for more efficient lookup)
        self.module_to_name_dict = {mod: name for name, mod in self.model.named_modules()}

    def register_parameter_hooks(self) -> list:
        """
        Registers hooks for parameter statistics collection in the model.

        This method sets up hooks to collect statistics for both forward and backward passes of the model's parameters.
        It supports models that are instances of `ParallelModule` and `InferenceModule`.
        For `ParallelModule`:
        - Computes forward statistics directly on the parameter.
        - Registers backward hooks to compute statistics during the backward pass if the parameter requires gradients.
        For `InferenceModule`:
        - Computes forward statistics directly on the parameter.
        Returns:
            list: A list of hooks registered for the model's parameters.
        """
        hooks = []
        if isinstance(self.model, ParallelModule):
            for _, p, m in self.model.named_parameters_with_meta():
                # NOTE: for forward, it is actually easier to directly compute statistics on the parameter instead of
                #       hooking it for execution.
                self.tensor_hook(
                    tensor=p.data,
                    name=f"parameter_statistics/layers.{m.layer_index}.{m.parameter_name}",
                    mode="forward",
                )
                # Backward hook
                if p.requires_grad:
                    hooks.append(
                        p.register_hook(
                            partial(
                                self.tensor_hook,
                                name=f"parameter_statistics/layers.{m.layer_index}.{m.parameter_name}",
                                mode="backward",
                            )
                        )
                    )

        elif isinstance(self.model, InferenceModule):
            for layer_idx, layer in enumerate(self.model._layers):
                for parameter_name, parameter in list(layer.named_parameters()) + list(layer.named_buffers()):
                    # NOTE: for forward, it is actually easier to directly compute statistics on the parameter
                    #       instead of hooking it for execution.
                    self.tensor_hook(
                        tensor=parameter.data,
                        name=f"parameter_statistics/layers.{layer_idx}.{parameter_name}",
                        mode="forward",
                    )

        return hooks

    @contextmanager
    def trace(self) -> Iterator[None]:
        """
        Context manager that registers hooks for tracking tensor statistics of a model.

        This method sets up hooks to monitor activations and gradients of the model's activations/parameters and
        custom tensors. For this, it registers parameter hooks, forward hooks, and backward hooks.
        The hooks are removed after the context is exited.

        Yields:
            None: This method is a generator that yields control back to the caller while the hooks are active.
        Raises:
            Any exception raised within the context will be propagated after the hooks are removed.
        """

        # Register parameter hooks
        parameter_hooks = self.register_parameter_hooks()

        # Register forward and backward module hooks
        forward_hook = torch.nn.modules.module.register_module_forward_hook(
            lambda module, input, output: self.activation_hook(module, input, output, mode="forward")
        )
        backward_hook = torch.nn.modules.module.register_module_full_backward_hook(
            lambda module, input, output: self.activation_hook(module, input, output, mode="backward")
        )

        try:
            with tensor_tracker.register_hook(self.tensor_hook):
                yield
        finally:
            # Remove hooks
            forward_hook.remove()
            backward_hook.remove()
            for hook in parameter_hooks:
                hook.remove()

    def _get_module_name(self, module: torch.nn.Module) -> Optional[str]:
        name = self.module_to_name_dict.get(module)  # quick lookup
        if name is None:
            # check if the module was later on added/modified in the model
            for other_name, other_module in self.model.named_modules():
                if module is other_module:
                    name = other_name
                    break
        return name

    def activation_hook(
        self,
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...] | torch.Tensor,
        output: tuple[torch.Tensor, ...] | torch.Tensor,
        mode: Literal["forward", "backward"] = "forward",
    ) -> None:
        """
        A hook function to record activation statistics for a given module during forward or backward pass.
        This function can be registered as a hook on a `torch.nn.Module` and will be called during the
        forward or backward pass. It computes and logs specified statistics for the module's output activations.
        Args:
            module (torch.nn.Module): The module to which this hook is registered.
            input: The input to the module. (unused)
            output (Union[torch.Tensor, tuple]): The output from the module.
                If a tuple, it is expected to contain a single tensor.
            mode (Literal["forward", "backward"], optional):
                The mode in which the hook is called, either "forward" or "backward". Defaults to "forward".
        Returns:
            None
        """
        name = self._get_module_name(module)
        if name is None:
            return
        if "_layers." in name:
            name_split = name.split(".")
            local_layer_idx = int(name_split[1])
            name = ".".join(name_split[2:])
        else:
            try:
                name_split = name.split("_")
                local_layer_idx = int(name_split[1])
            except (IndexError, ValueError):  # cannot automatically infer local_layer_dix
                local_layer_idx = -1

        if local_layer_idx < 0:
            layer_idx: int | str = "index_not_found"
        elif hasattr(self.model, "_pipe_partition_coordinates"):
            layer_idx = local_layer_idx + self.model._pipe_partition_coordinates[0].start
        else:
            layer_idx = local_layer_idx

        if self.config.include_module_type:
            name = f"activation_statistics/layers.{layer_idx}.{type(module).__name__}.{name}"
        else:
            name = f"activation_statistics/layers.{layer_idx}.{name}"

        # Get statistics to be computed for this module name
        statistics_ftcs = statistics_fcts_for_name_pattern(self.config.statistics, name)
        if statistics_ftcs is None:
            return

        if isinstance(output, tuple):
            if len(output) != 1:
                self.logger.warning(
                    f"got a tuple of length {len(output)}, therefore skipping {mode} activation hook for {name}"
                )
                return
            output = output[0]

        if not isinstance(output, torch.Tensor):
            self.logger.warning(
                f"got a value of type {type(output)}) but can only handle torch.Tensor, "
                f"therefore skipping {mode} tensor hook for {name}"
            )
            return

        metrics = {(name + "_" + mode + "_" + s): fn(output) for s, fn in statistics_ftcs.items()}
        self.logger.log_metrics(metrics=metrics, step=self.context.iterations, to_info=False, to_wandb=False)

    def tensor_hook(self, tensor: torch.Tensor, name: str, mode: Literal["forward", "backward"]) -> None:
        """
        Hook function to record statistics of a given tensor/parameter during forward or backward pass.

        Args:
            tensor (torch.Tensor): The tensor for which statistics are to be recorded.
            name (str): The name of the parameter.
            mode (Literal["forward, backward"]):
                The mode indicating whether the hook is for the forward or backward pass.
        Returns:
            None
        """
        # Get statistics to be computed for this parameter name
        statistics_ftcs = statistics_fcts_for_name_pattern(self.config.statistics, name)
        if statistics_ftcs is None:
            return

        if isinstance(tensor, tuple):
            if len(tensor) != 1:
                self.logger.warning(
                    f"got a tuple of length {len(tensor)}, therefore skipping {mode} tensor hook for {name}"
                )
                return
            tensor = tensor[0]

        if not isinstance(tensor, torch.Tensor):
            self.logger.warning(
                f"got a value of type {type(tensor)}) but can only handle torch.Tensor, "
                f"therefore skipping {mode} tensor hook for {name}"
            )
            return

        metrics = {(name + "_" + mode + "_" + s): fn(tensor) for s, fn in statistics_ftcs.items()}
        self.logger.log_metrics(metrics=metrics, step=self.context.iterations, to_info=False, to_wandb=False)
