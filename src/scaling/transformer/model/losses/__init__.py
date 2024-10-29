from scaling.transformer.context.config import (
    BaseLossFunctionConfig,
    ContrastiveLossFunctionConfig,
    CrossEntropyLossFunctionConfig,
    LossFunctionType,
)
from scaling.transformer.context.context import TransformerContext

from .contrastive import ContrastiveLoss
from .cross_entropy import CrossEntropyLoss


def create_loss_function(
    context: TransformerContext, loss_function_config: BaseLossFunctionConfig
) -> ContrastiveLoss | CrossEntropyLoss:
    match loss_function_config.loss_type:
        case LossFunctionType.CONTRASTIVE_LOSS:
            assert isinstance(
                loss_function_config, ContrastiveLossFunctionConfig
            ), "Provide ContrastiveLossFunctionConfig when selecting contrastive_loss"
            return ContrastiveLoss(topology=context.topology, loss_config=loss_function_config)
        case LossFunctionType.CROSS_ENTROPY_LOSS:
            assert isinstance(
                loss_function_config, CrossEntropyLossFunctionConfig
            ), "Provide CrossEntropyLossFunctionConfig when selecting contrastive_loss"
            return CrossEntropyLoss(
                loss_config=loss_function_config, umup_config=context.config.transformer_architecture.umup
            )
        case _:
            raise ValueError(f"Unknown loss function type: {loss_function_config.loss_type}")
