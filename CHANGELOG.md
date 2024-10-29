# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-10-29

### Changed
- Use poetry with pip instead of conda with pip for env management
- Upgraded to PyTorch 2.4
- Renamed `allowed_missing_keys_in_optimizer_groups` was renamed to `allow_missing_params_in_optimizer`

### Removed
- Removed `finetune` from training config. This field is replaced by optimizer groups
- Removed `finetunable_parameters` from training config. This field is replaced by optimizer groups
- Removed `parameters_exclude` from training config. Those fields are replaced by optimizer groups
- Removed `use_separate_lr_on_embeddings` from training config. Those fields are replaced by optimizer groups

### Added
- Implemented U-MUP method
- Implemented FP8 linear layers for training and inference (naive casting, no per-tensor-scaling)
- Tensor Statistics Recorder for monitoring activation and parameter distributions
- Configurable Loss Functions
- Configurable Language Modeling Head
- Added Cross Entropy Loss as Configurable Loss
- Added Contrastive Loss as Configurable Loss
- Added Memory Map Dataset based on Protobuf serialization
- Semantic Embedding Inference
- Semantic Embedding Inference Example
- Added `training_groups` for configurable optimizer groups
- Added tests for Transformer example and MLP example

### Fixed
- Fix Pydantic Warning on Startup


## [0.1.0] - 2024-08-22

### Added
- Added core and transformer modules
