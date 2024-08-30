```
ALEPH ALPHA
┏┓   ┓•
┗┓┏┏┓┃┓┏┓┏┓
┗┛┗┗┻┗┗┛┗┗┫
          ┛
```
#

Scaling is a distributed training library and installable dependency designed to scale up neural networks, with a dedicated module for training large language models.

# Features

Scaling consists of two primary components, a model-agnostic core module (`scaling.core`), which functions as the engine for distributed training workloads, and the `scaling.transformer` suite, specifically designed for LLM training.
The Scaling core module features various parallelization and partitioning techniques:

- __Data parallelism__: Distribute training data across multiple devices.
- __Pipeline parallelism__: Divide model layers into sequential stages across multiple devices.
- __Tensor parallelism__: Split individual tensors across multiple devices.
- __3D parallelism__: Seamlessly combine data, pipeline, and tensor parallelism.
- __ZeRO sharding__: Support for optimizer state partitioning (ZeRO-1) in data parallel training regimens.
- __Efficient training__: Support for modern performance optimizations such as mixed precision training and activation checkpointing.
- __Code quality standards__: Rigorous typing, Pydantic classes and extensive tests for ease of development and less potential for bugs.

Built upon the Scaling core components, the Transformer module implements a state-of-the-art transformer architecture and training loop.
Among the featured architecture options we support:

- Multi-query and grouped-query attention,
- Different MLP types (e.g., SwiGLU),
- Rotary positional embeddings,
- Parameter-efficient fine-tuning methods: Bitfit, Adapters, LoRA.

# Getting Started

## Installation

The installation requires Linux with Python 3.10 and PyTorch 2.1.1.
You will also need the appropriate CUDA dependencies and version installed on your system for GPU support.
Clone this repository and install:

```bash
pip install .
```

### Flash Attention

To install Flash Attention, you need to make sure you have PyTorch installed already. 
Simply install the base depenendencies with `pip install .` before installing Flash Attention.
Then install Flash Attention with:

```bash
pip install .[gpu_optimization]
```

Ensure that your environment variables are set correctly.
The `CUDA_HOME` variable should point to the location of your CUDA installation.
For additional information or troubleshooting, please refer to [the official documentation](https://github.com/Dao-AILab/flash-attention).
You can then use Flash Attention in your Transformer architecture configuration:

```json
{
    "transformer_architecture": {
        ...
        "masked_softmax": {
            "kernel": "flash_attention",
            "softmax_in_fp32": true,
            "deterministic_flash_attn_bwd": false,
            "scale": 1.0,
        },
        ...
    }
}
```

## Quick Start

Everything you need to start a full distributed transformer training is contained in [this example](/examples/transformer_example/).

You can start a training job by executing:

```bash
python3 -m examples.transformer_example.run examples/transformer_example/config.yml
```

Feel free to experiment with the [example config](/examples/transformer_example/config.yml) that controls all relevant training parameters and modify it to suit your needs.
In particular, update the topology configuration to reflect the amount of GPU devices available.
For instance, if you have a single GPU device available, set the topology parameters accordingly.

```json
{
    ...
    "topology": {
        ...
        "model_parallel_size": 1,
        "pipe_parallel_size": 1,
        "data_parallel_size": 1,
        ...
    }
}
```

Note: The number of available GPU devices needs to be equal to ```model_parallel_size * pipe_parallel_size * data_parallel_size```. To control this, you can simply set the ```CUDA_VISIBLE_DEVICES``` environment variable to the desired GPU indices.

If you want to run a large-scale job on a cluster, you can in principle use the same code, but make sure the training script gets executed in parallel. We also provide the tooling to do this. For an in-depth look, check out our more detailed guide on [how to train a model on multiple nodes](/examples/tutorials/multi_node_training.md).
Scaling also features [a basic inference module](/examples/tutorials/inference.md) to generate outputs from model checkpoints.

### A 3D Parallel MNIST MLP Training

If you are interested to learn more about how to build your own training library using Scaling, check out our [MLP example](/examples/mlp_example/).
The MNIST MLP classifier is probably the most used example across myriads of Deep Learning tutorials, and Scaling makes no exception.
This is a self-contained codebase built on Scaling that implements a full 3D parallel training loop for MNIST classification that is as simple as it gets while touching upon all important components in ```scaling.core```.
At the end of the day, our transformer training suite ```scaling.transformer``` is built in the very same fashion.
The MLP example is the best way to start if you want to learn about how to use the building blocks from ```scaling.core``` without getting lost in the details of a complex model architecture.

# Development

Additional dependencies are required if you want to run tests or type checks.
Install them as follows:

```bash
pip install -e .[test]
```

## Mypy

Run mypy to catch typing mistakes:

```
mypy src
mypy tests
```

## Tests

Run tests with:

```
pytest tests/core
pytest tests/transformer
```
