from typing import Any, List, Type

import torch

from .base_layer import BaseLayer


class LayerSpec:
    def __init__(self, module_class: Type[BaseLayer], **kwargs: Any) -> None:
        super().__init__()

        self.module_class = module_class
        self.kwargs = kwargs

    def initialize(self, device: torch.device | None = None) -> BaseLayer:
        return self.module_class(**self.kwargs).cuda(device=device)


class TiedLayerSpec(LayerSpec):
    def __init__(
        self,
        key: str,
        tied_weight_attributes: List[str],
        module_class: Type[BaseLayer],
        **kwargs: Any,
    ) -> None:
        super().__init__(module_class=module_class, **kwargs)
        self.key = key
        assert len(set(tied_weight_attributes)) == len(
            tied_weight_attributes
        ), f"duplicates in tied_weight_attributes: {tied_weight_attributes}"
        self.tied_weight_attributes = tied_weight_attributes
