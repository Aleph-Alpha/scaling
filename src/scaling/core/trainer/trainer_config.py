from pathlib import Path
from typing import List, Optional

from pydantic import Field

from ..config import BaseConfig


class TrainerConfig(BaseConfig):
    save_dir: Optional[Path] = Field(None, description="directory for saving checkpoints")
    save_interval: Optional[int] = Field(
        None,
        description="save a checkpoint every 'save_interval' steps to save_dir, iff save_dir is defined",
    )

    load_dir: Optional[Path] = Field(None, description="directory for loading checkpoints")

    train_iterations: Optional[int] = Field(
        None,
        description="train for this number of iterations",
    )

    assert_checkpoint_loaded: bool = Field(
        True,
        description="error out if a checkpoint could not be loaded",
    )

    load_optimizer_states: bool = Field(True, description="load optimizer states on checkpoint load")

    delete_past_optimizer_states: bool = Field(
        True,
        description="Deletes optimizer states on the last n-1 checkpoints right after saving the nth checkpoint",
    )

    load_context: bool = Field(
        True,
        description="load context state, i.e. train iterations, consumed train and eval samples on checkpoint load",
    )

    allowed_missing_keys_in_checkpoint: Optional[List[str]] = Field(
        None,
        description="list of parameter names that may not be present in an existing checkpoint. "
        "This is helpful for e.g. adapters initialization.",
    )

    allowed_unexpected_keys_in_checkpoint: Optional[List[str]] = Field(
        None,
        description="list of parameter names that may be present in an existing checkpoint but not be loaded. "
        "This is helpful for e.g. getting rid of finetunings without editing the checkpoint manually.",
    )

    ignore_keys_in_checkpoint: Optional[List[str]] = Field(
        None,
        description="list of parameter names for which we do not want to load the pretrained weights. "
        "Use this if you want to reinitialize parts of a pretrained model.",
    )

    merge_lora_after_loading_checkpoint: Optional[bool] = Field(
        False,
        description="This needs to be set in order to merge LoRa weights after loading",
    )

    seed: int = Field(42, description="")

    dataloader_num_workers: int = Field(0, description="")
    dataloader_pin_memory: bool = Field(True, description="")
    dataloader_prefetch_factor: Optional[int] = Field(None, description="")

    eval_iterations: int = Field(
        1,
        description="evaluate on this number of iterations every eval_iterations steps (Currently not Implemented)",
    )

    eval_interval: Optional[int] = Field(
        None,
        description="evaluate every eval_iterations steps",
    )

    separate_file_for_parameters: Optional[List[str]] = Field(
        None,
        description="create a separate checkpoint file for parameters matching these names",
    )
