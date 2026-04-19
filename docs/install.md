# Install

This page covers environment setup only. For usage, go to [Quick Start](getting_started/quick_start.md) or the [Workflow](workflow/data_format.md) section.

## Requirements

- Python 3.12
- CUDA 12 or newer for GPU-backed training and inference
- `nvcc` available when compiling the `rspmm` kernel
- Poetry if you plan to install from source or contribute to the repository

## Install With Conda

```bash
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.6.3
pip install gfmrag
```

## Install With Pip

```bash
pip install gfmrag
```

## Install From Source

```bash
git clone https://github.com/RManLuo/gfm-rag.git
cd gfm-rag
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.6.3
poetry install
```

## Optional LLM Backends

### Llama.cpp

```bash
pip install llama-cpp-python
```

References:

- <https://python.langchain.com/docs/integrations/chat/llamacpp/>
- <https://github.com/abetlen/llama-cpp-python>

### Ollama

```bash
pip install langchain-ollama
pip install ollama
```

Reference:

- <https://python.langchain.com/docs/integrations/chat/ollama/>

## Troubleshooting

### CUDA errors when compiling `rspmm`

If compilation fails, make sure `nvcc` is available and `CUDA_HOME` points to the installed toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
```

If CUDA was installed via conda, `CUDA_HOME` is often configured automatically.

### `rspmm` compilation appears stuck

Clear the extension cache and retry:

```bash
rm -rf ~/.cache/torch_extensions
```

### Need development commands instead of installation

See [Development](DEVELOPING.md) for `mkdocs`, `pre-commit`, and packaging commands.
