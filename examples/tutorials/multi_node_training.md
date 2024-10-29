# Docker and Multi-Node Training

This guide walks you through how to train models using Scaling within Docker containers and in a multi-node setup, end-to-end.
It focuses on the technical aspects of how to start a training run.
Scaling supports a variety of means to allow training on different infrastructures.
These include:

* Training (small) models locally on a single node with Docker
* Training models on multiple nodes with `pdsh` and Python
* Training models on multiple nodes with `pdsh` and Docker

This guide mostly aims at enabling large-scale training using our ```scaling.transformer``` codebase, but most of the content applies more generally to any training loop that is built on ```scaling.core```.

Note that this guide assumes no explicit cluster management tool, such as Determined or Ray, is used.

## Configuring a Model Training

Generally, Scaling provides a [Pydantic](https://docs.pydantic.dev/latest/)-based configuration system to configure a model for training.
For transformer training, we provide an [example configuration](/examples/transformer_example/config.yml).
This configuration encompasses the following top-level keys:

* `runner`: How to run the training. Concerns tooling such as pdsh, Docker, or determined.
* `logger`: Where to send logs and metrics. Supports wandb, tensorboard, and determined.
* `topology`: How to partition and distribute the model training on a cluster.
* `optimizer`: Configure the optimizer, e.g. AdamW.
* `learning_rate_scheduler`: Configure the learning rate, decay and warmup.
* `trainer`: Configure the training loop. This covers the number of iterations, how often to save, what to load, all the info required to load a checkpoint during training, etc.
* `profiler`: Collect profiling data on the training performance.
* `transformer_architecture`: Configure the transformer architecture (hidden size, number of layers, etc.).
* `data`: Where to load the training data from.

## Single-Node Training with Docker

You can start a training with Docker on a single node.
To use this method, you will need to create a container image with your training code based on Scaling.
Next, create a copy of the [example configuration](/examples/transformer_example/config.yml) and adjust the following fields:

```yaml
{
    "runner": {
        # The runner type is "pdsh_docker": if the "hosts" field is null, this will not invoke pdsh,
        # but instead launch the training on the current node.
        "runner_type": "pdsh_docker",

        "hosts": null,

        # Set the train script to run *inside* the container, for transformer training it needs to be set to this one:
        "script": "src/scaling/transformer/train.py",

        "docker_config": {
            # Set the container image to use
            "docker_container": "transformer_example_image",
            # ...
        }
    },

    # ...
}
```

Then, from the root of the repository, you can start the training as follows:

```sh
python3 examples/transformer_example/run.py path/to/your/configuration.yml
```

## Multi-node Training with PDSH

Scaling supports distributed training with pdsh.
pdsh is a simple tool to issue commands to groups of hosts in parallel.
It will start and configure instances of the training software on the cluster.
The training software can either directly run on the hosts with Python or within Docker containers.

### Python

To directly run the training software on the cluster hosts, you will need to set up a Python environment with Scaling, the training code and the dependencies on all the nodes in the cluster.
Refer to the top-level [README.md](/README.md) for more information.
Unless you have a different means of automating this deployment, it is recommended to use a Docker-based setup instead (see below).

With the infrastructure in place, you can create the model training configuration.
For that, create a copy of the [example configuration](/examples/transformer_example/config.yml) and adjust the following runner fields:

```yaml
{
    "runner": {
        # Set the runner mode to use plain pdsh
        "runner_type": "pdsh",

        # Configure the hosts and the GPU slots to use
        "hosts": [
            "host-01 slots=1",
            "host-02 slots=1",
            ...
        ],

        # Script on the hosts to execute ,for transformer training it needs to be set to this one:
        "script": "/opt/transformer_example/src/transformer_example/train.py",

        # ...
    },

    # ...
}
```

Next, you need to choose a machine that has SSH access to all the nodes in the cluster.
This is because `pdsh` will use SSH in the background to launch the training software.
Finally, from that node, launch your model training configuration as follows:

```sh
python3 examples/transformer_example/run.py path/to/your/configuration.yml
```

### Docker

To use this method, you will need to create a container image with your training code based on Scaling.
In the cluster, make sure this container image is available to all machines that should run the training software.
Usually, this will involve something like:

```sh
docker login -u token registry.example.com   # where registry.example.com is your image registry
docker pull transformer_example_image           # where transformer_example_image is your training code image
```

Further you need to make sure all machines in the cluster have network connectivity with each other, otherwise the training will run into connectivity errors.
Notice that NCCL will try to automatically detect the network interface to use.
If this fails, this [can be configured](https://pytorch.org/docs/stable/distributed.html#choosing-the-network-interface-to-use).

With the infrastructure in place, you can now create the model training configuration.
For that, create a copy of the [example configuration](/examples/transformer_example/config.yml) and adjust the following runner fields:

```yaml
{
    "runner": {
        # Set the runner mode to use pdsh with Docker
        "runner_type": "pdsh_docker",

        # Configure the hosts and the GPU slots to use
        "hosts": [
            "host-01 slots=1",
            "host-02 slots=1",
            ...
        ],

        # Set the train script to run *inside* the container, for transformer training it needs to be set to this one:
        "script": "src/scaling/transformer/train.py",

        "docker_config": {
            "docker_container": "transformer_example_image",
            # ...
        },

        # ...
    },
}
```

Next, you need to choose a machine that has SSH access to all the nodes in the cluster.
This is because `pdsh` will use SSH in the background to launch the training software.
Finally, from that node, launch your model training configuration as follows:

```
python3 examples/transformer_example/run.py path/to/your/configuration.yml
```

## Mounting Directories within Containers

Once you go beyond the transformer example, you will likely want to load your own data, as well as retain model checkpoints. When training with Docker using PDSH or otherwise, you will have to mount the desired directories to load the data from, respecively save the checkpoints to. This can be done by adjusting the following fields in the model training configuration:

```yaml
{
    "runner": {
        "docker_config": {
            "docker_mounts": [
                [
                    # Path on the host
                    "/path/on/host/to/dataset",
                    # Path inside the container
                    "/dataset",
                    # Path on the host
                    "/path/on/host/to/checkpoints",
                    # Path inside the container
                    "/checkpoints"
                ],
            ],
            # ...
        },
        # ...
    },
    # ...
    "data": {
        "data_prefixes": ["/dataset"],
    },
    "trainer": {
        # Path inside the container
        "save_dir": "/checkpoints",

        # Note that you will also need to set the "save_interval",
        # The training code will save a checkpoint every "save_interval" steps.
        "save_interval": 100,
    }
    # ...
}
