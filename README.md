

<div align="center">
  <h1 style="font-size: 40px;">AlignXplore Plus</h1>
  <p>A Framework for Transferable Personalization with the Textual Interface.</p>
</div>

## 1. Introduction

AlignXplore Plus is a framework for transferable personalization with the textual interface. It moves to natural language as a universal, model- and task-agnostic interface for user representation. It is a two-stage training framework with supervised fine-tuning and reinforcement learning. More details can be found in the full [paper](https://arxiv.org/abs/xxxx.xxxxx).

## 2. Links

- 📜 [Paper](https://arxiv.org/abs/xxxx.xxxxx)
- 🤗 [Model](https://huggingface.co/xxxxx)
- 🤗 [Dataset](https://huggingface.co/datasets/xxxxx)

## Requirments

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## Training


```bash
cd verl

bash examples/grpo_trainer/run_streaming_8p.sh
```

## Inference

```bash
cd eval

# You can modify line #21 to the path of your model.
bash gen_preference.sh
bash straming_gen_preference.sh
# You should modify line #21 to the model name of your model.
bash evaluation.sh
```

## Citation

```bibtex
@article{xxxxx,
  title={},
  author={xxxxx},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
```