# Inference

Scaling includes a basic inference module to generate outputs from model checkpoints.
You can try it out by [downloading the Pharia-1 checkpoints from Hugging Face](https://huggingface.co/Aleph-Alpha/Pharia-1-LLM-7B-control).
The Hugging Face repository contains everything necessary to load the model, simply pass the path to the repository to the inference module:

```python
from pathlib import Path

from scaling.transformer.inference import TransformerInferenceModule

inference_model = TransformerInferenceModule.from_checkpoint(
    checkpoint_dir=Path("path/to/model-checkpoint"),
)

input_text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|><|start_header_id|>user<|end_header_id|>

When was Rome founded?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


"""

generation = inference_model.generate(max_tokens=100, input_text=input_text)
print(generation.completion_text)
```
