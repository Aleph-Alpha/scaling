# MLP Example

This is a minimal example and walkthrough on how to declare and train a parallelized model using Scaling.
For a more comprehensive example, implementing a full transformer architecture, please refer to the [transformer submodule](/src/scaling/transformer).

## Prerequisites

To install the required dependencies for this walkthrough, please follow the installation instructions provided in the [top-level README](/README.md) of this repository.

## Walkthrough

Similar to a typical training loop, we need to set up various components, such as a PyTorch module to define the model, an optimizer, and data loaders, to run the training.
At the top level, Scaling provides a `BaseTrainer` class that [simplifies the setup of these parallelized training loops](/src/scaling/core/trainer/trainer.py#L33) for us.

```python
trainer = BaseTrainer(
    config=...,
    context=...,
    dataset=...,
    dataset_evaluation=...,
    parallel_module=...,
    optimizer=...,
    sync_batch_to_model_parallel=...,
    metrics_aggregation_fn=...,
    loss_function=...,
)

trainer.run_training()
```

Ultimately, to start a distributed training, the following steps are needed:
1. Create a training script ([here](/examples/mlp_example/train.py) ```train.py```) that instantiates the trainer and starts the training loop via ```trainer.run_training()```.
2. Create a run script ([here](/examples/mlp_example/run.py) ```run.py```) that starts the training script in multiple processes with the correct environment variables to establish parallelized training based on the topology specified in the configuration.
This may sound complicated, but the multi-processing logic is taken care of by the function `runner_main` that is contained in `scaling.core` (note that, depending on your cluster setup, a custom multi-process launcher might be required).
The run script then looks like this:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = MLPConfig.from_yaml(args.config)
    logger.configure(config=config.logger, name="runner")
    runner_main(config.runner, payload=config.as_dict())
```

The subconfig ```config.runner``` contains the field ```script``` that points to the training script to execute.

To configure custom options for parameterizing the model (`MLPConfig` in this example) you wish to train, you can extend the `BaseConfig` model and add your configuration options under [a new key that you define](/examples/mlp_example/config.py#L66).
Subsequently, create a context class extending `BaseContext` to [hold both the topology and configuration](/examples/mlp_example/context.py#L7).

To see this example in action, start a training by running the command from the root directory of Scaling:

```bash
python3 -m examples.mlp_example.run examples/mlp_example/config.yml
```

### 1. Creating a Parallel Module

The `ParallelModule` class is the core of Scaling and is used to declare a model for parallelization.
A `ParallelModule` takes a list of `LayerSpec` objects, which contain the specifications of the layers in the model, and instantiates them.
The `LayerSpec` objects hold references to the module classes we want to instantiate along with their arguments. The second argument for `ParallelModule`is a `Topology`object that contains all information on the device configuration and how to shard the model.

```python
layer_specs = [
    LayerSpec(
        module_class=MLPLayerRowParallel,
        topology=context.topology,
        in_features=784,
        out_features=64,
    ),
    LayerSpec(
        module_class=MLPLayerColumnParallel,
        topology=context.topology,
        in_features=64,
        out_features=10,
    ),
]

parallel_module = ParallelModule(
    layer_specs=layer_specs,
    topology=topology,
)
```

In the following example, we define a module that implements a simple linear layer with column-wise parallelism.
This module uses the `ColumnParallelLinear` module to distribute the linear transformation across multiple devices according to the specified topology.

```python
class MLPLinearColumnParallel(MLPBaseLayer):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        topology: Topology,
    ):
        super().__init__()
        self.linear = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            parallel_output=True,
            topology=topology,
        )

    def forward(self, x: MLPLayerIO) -> MLPLayerIO:
        preactivations = torch.flatten(x.activations, start_dim=1)
        preactivations = self.linear(preactivations)
        activations = f.relu(preactivations)
        return MLPLayerIO(activations=activations)
```

Each module must [extend a class that implements the `BaseLayer` class](), which is a [generic abstract class](/src/scaling/core/nn/parallel_module/base_layer.py#L16) designed to standardize the implementation of layers in the neural network.
The `BaseLayer` class inherits from `torch.nn.Module` and implements a generic interface for the input, output, and last layer output types, and enforces a consistent interface for converting inputs and outputs.

### 2. Initializing the Optimizer

To facilite the optimizer setup for parallel training, we provide the `Optimizer` and `OptimizerParamGroup` wrappers.
`OptimizerParamGroup` handles mixed-precision training and zero-sharding across parallel devices.
A parameter group configuration specifies additional settings such as weight decay and a learning rate schedule for a group.

```python
parameter_groups = [
    OptimizerParamGroup(
        named_parameters_with_meta=[
            (n, p, m) for n, p, m
            in parallel_module.named_parameters_with_meta()
        ],
        config=OptimizerParamGroupConfig(
            name="weight_decay_params",
            weight_decay=context.config.training.weight_decay,
            learning_rate_scheduler=context.config.learning_rate_scheduler,
        ),
    ),
]

optimizer = Optimizer(
    config=context.config.optimizer,
    parameter_groups=parameter_groups,
    topology=context.topology,
)
```

### 3. Loss Function and Metrics Collection

The `loss_function` processes the output from the final layer of the parallel module.
It returns two items: the loss, which is used by the trainer for backpropagation, and a dictionary of customizable metric data that we want to keep track of.
Using a `metrics_aggregation_fn`, you can [define how the metrics need to be aggregated](/examples/mlp_example/model.py#L120) across ranks.

```python
def loss_function(
    output: MLPLayerIO,
    batch: MNISTDatasetBatch,
) -> tuple[torch.Tensor, dict[str, float]]:

    loss = torch.nn.functional.cross_entropy(
        output.activations,
        torch.tensor(batch.targets, dtype=torch.long)
    )

    accuracy = (
        sum(output.activations.argmax(dim=1) == batch.targets)
        / batch.targets.shape[0]
    )

    return loss, { "accuracy": accuracy }
```

### 4. Datasets

The dataset classes provide a structured way of handling datasets in a parallelized training regime.
Each dataset class must extend the `BaseDataset` class, a generic abstract class that enforces a consistent interface for dataset management.
This class inherits from `torch.utils.data.Dataset` and [requires implementation of several methods](/examples/mlp_example/data.py#L45).
These methods handle tasks such as identifying the dataset, returning its length, retrieving items by index, setting the seed for shuffling, and collating a batch of dataset items, respectively.

Additionally, the `sync_batch_to_model_parallel` method synchronizes a batch to a model parallel topology, which must also be implemented in child classes.
The `BaseDatasetItem` and `BaseDatasetBatch` classes serve as base classes for individual dataset items and batches, respectively.
Methods `only_inputs` and `only_targets` in `BaseDatasetBatch` must be implemented to reduce memory overhead by removing non-essential properties from the batch.

### 5. Putting Everyting Together

Now that all the components are in place, you can instantiate a `BaseTrainer` and start your training.
Refer to `train.py` to see [all the pieces tied together](/examples/mlp_example/train.py).

Good luck on your Scaling journey making your datacenter sweat - what could possibly go wrong?
