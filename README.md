

<div align="center">
  <h1 style="font-size: 40px;">AlignXplore+</h1>
  <p>A Framework for Transferable Personalization with the Textual Interface.</p>

  
  [![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.18071)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SFT%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-SFT)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-RL%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-RL)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Eval%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-Benchmark)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/VanillaH1/AlignXplore-Plus)

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

## SFT Training

You should first  download the dataset from [here](https://huggingface.co/datasets/xxxxx), then generate the tokenized dataset by running the following script.

```bash
cd sft

python prepare_dataset.py
```

You can modify line 21 and line 34 to set the path to your own model and tokenized dataset.

```python
> sft/sft.py

21  model_name_or_path = "Qwen/Qwen3-8B" 

34  dataset = load_from_disk("tokenized_dataset")
```

Then set the node address and other distrubuted training parameters in the following script.

```bash
cd sft

bash sft.sh
```

## RL Training 

You should first download the RL dataset from [here](https://huggingface.co/datasets/xxxxx) and run the following script to generate verl-format dataset.

```bash
cd verl

# set line 49 data_source_train = "path to jsonl data" to your downloaded dataset path.

python example/data_preprocess/upi_streaming_dataset.py
```

Then you can set your own path and run the RL training in the following script.

```bash
cd verl

bash examples/grpo_trainer/run_streaming_8p.sh
```

## Inference

```bash
cd eval

# You should modify line #21 to the path of your model.
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
