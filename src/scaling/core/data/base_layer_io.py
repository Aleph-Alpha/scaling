import torch


class BaseLayerIO:
    def to_(self, device: torch.device) -> None:
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.to(device))

    def contiguous_(self) -> None:
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.contiguous())
