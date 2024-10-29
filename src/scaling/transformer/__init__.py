import os

from .context import TransformerConfig, TransformerContext
from .data import (
    EmbeddingBlendedDataset,
    EmbeddingDataset,
    FinetuningChatBlendedDataset,
    FinetuningChatDataset,
    FinetuningTextBlendedDataset,
    FinetuningTextDataset,
    LegacyBlendedDataset,
    TextBlendedDataset,
    TextDataset,
    TextDatasetItem,
)
from .model import TransformerLayerIO, TransformerParallelModule, init_model, init_optimizer

# This will help us to debug some determined problems, and also is the entrypoint
# that most likely every research repo gets into.
det_variables = {k: v for k, v in os.environ.items() if k.startswith("DET_")}
print(f"DET_ENV_VARS: {det_variables}", flush=True)
