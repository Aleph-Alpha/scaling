import torch

from scaling.core import LayerNorm, LayerNormConfig

from .clip import ClipModifiedResNet


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        out_features: int,
        device: torch.device,
        dropout_p: float = 0.0,
        layernorm_config: LayerNormConfig | None = None,
        image_encoder: str = "ClipRN50x16",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        assert image_encoder == "ClipRN50x16", "only clip implemented"
        self.input_encoder_output_dim = 3072
        self.input_encoder = (
            ClipModifiedResNet(
                layers=[6, 8, 18, 8],
                num_init_channels=96,
            )
            .to(dtype)
            .to(device)
        )
        self.image_encoder_image_size = (384, 384)

        down_sample_size = 32
        assert down_sample_size is not None
        num_encoder_tokens = (self.image_encoder_image_size[0] // down_sample_size) * (
            self.image_encoder_image_size[1] // down_sample_size
        )

        image_encoder_num_embedding_tokens = num_encoder_tokens
        num_token_stack = num_encoder_tokens // image_encoder_num_embedding_tokens

        self.do_token_reshape = num_token_stack != 1
        self.input_encoder_output_dim = self.input_encoder_output_dim * num_token_stack
        self.reshape_shape = (
            image_encoder_num_embedding_tokens,
            self.input_encoder_output_dim,
        )

        self.proj = torch.nn.Linear(
            self.input_encoder_output_dim,
            out_features,
            device=device,
            dtype=dtype,
        )
        self.dropout = torch.nn.Dropout(dropout_p)

        if layernorm_config is not None:
            self.has_layernorm = True
            self.layernorm = LayerNorm(
                config=layernorm_config,
                normalized_shape=out_features,
                device=device,
                dtype=dtype,
            )
        else:
            self.has_layernorm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_encoder(x)

        if self.do_token_reshape:
            batch_size = x.size(dim=0)
            new_shape = (batch_size,) + self.reshape_shape
            x = torch.reshape(input=x, shape=new_shape)

        x = self.proj(x)
        x = self.dropout(x)
        if self.has_layernorm:
            x = self.layernorm(x)
        return x
