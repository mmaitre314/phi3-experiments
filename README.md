# Phi-3 Experiments

Kicking the tires of [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/).

## Getting Started

For all experiments, install:
- [Miniconda](https://docs.anaconda.com/free/miniconda/)
- [VSCode](https://code.visualstudio.com/Download)
  - Extensions [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

Then follow experiment-specific install steps below.

### Hugging Face

Install [CUDA 12](https://developer.nvidia.com/cuda-downloads)

Open a shell and run:
```bash
conda env create -f env-hf.yaml
conda activate phi3-hf
code .
```
Then open the notebook `hf.ipynb`.

### ONNX

#### CUDA

Install:
- [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN 8.9.2.26](https://developer.nvidia.com/rdp/cudnn-archive)

> TODO: try to pip install CUDA + cuDNN instead

Open a shell and run:
```bash
conda env create -f env-onnx-cuda.yaml
conda activate phi3-onnx-cuda
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cuda/cuda-int4-rtn-block-32/* --local-dir .
code .
```
Then open the notebook `onnx-cuda.ipynb`.

#### DirectML

Open a shell and run:
```bash
conda env create -f env-onnx-dml.yaml
conda activate phi3-onnx-dml
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include directml/* --local-dir .
code .
```
Then open the notebook `onnx-dml.ipynb`.

#### CPU

Open a shell and run:
```bash
conda env create -f env-onnx-cpu.yaml
conda activate phi3-onnx-cpu
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
code .
```
Then open the notebook `onnx-cpu.ipynb`.

## Experiments

Hardware:
- GPU: GeForce GTX 1650 Ti (8GB)
- CPU: Intel Core i5 10300H (4 cores, 8GB)

Experiment | Status
--|--
Hugging Face | `OutOfMemoryError` on GPU, super slow on CPU
ONNX CUDA | Basic generation running
ONNX DirectML | `ImportError: DLL load failed while importing onnxruntime_genai`
ONNX CPU | Basic generation running

Benchmark for 'Tell a joke' prompt:

Experiment | Wall time
--|--
Hugging Face | -
ONNX CUDA | 2.7s
ONNX DirectML | -
ONNX CPU | 12.0s

## References

- https://huggingface.co/docs/transformers/main/model_doc/phi3
- https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html
- ONNX requirements https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
- ONNX GenAI script phi3-qa.py https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3-qa.py