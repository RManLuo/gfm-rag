# Changelog

## 0.2.2
- Fix: load model from pre-trained with different architectures
- Release the [full training data](https://drive.google.com/drive/folders/11iTxDWtECnkGdiCkMlp0Mh2MFqItdvcY?usp=drive_link).

## 0.2.1
- Fix `colbert-ai` dependency with new version of `torch`.

## 0.2.0
- A new version of [GFM-RAG (2025-06-03)](https://huggingface.co/rmanluo/GFM-RAG-8M/commit/62cf6398c5875af1c4e04bbb35e4c3b21904d4ac) which is pre-trained on 286 KGs.

|                      | HotpotQA |          | MuSiQue  |          | 2Wiki    |          |
|----------------------|----------|----------|----------|----------|----------|----------|
| Models               | R@2      | R@5      | R@2      | R@5      | R@2      | R@5      |
| GFM-RAG (2025-02-06) | 78.3     | 87.1     | 47.8     | 58.2     | 89.1     | 92.8     |
| GFM-RAG (2025-06-03) | **81.5** | **89.6** | **50.0** | **59.3** | **90.1** | **93.6** |

- Add on-demand data loader to support training on large-scale datasets.

## 0.1.3
- Add NVIDIA backend for LLM api
- Fix Ollama Error in OpenIE extraction
- Replace torch_scatter with native torch scatter
- Improve documentation

## 0.1.2
- Fix typos
- Remove example data from git

## 0.1.1
- Fix typos
- Fix pypi publish

## 0.1.0
- Initial release
