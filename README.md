

<div align="center">
  <h1 style="font-size: 40px;">AlignXplore+</h1>
  <p>Text as a Universal Interface for Personalizing Large Language Models</p>

  
  [![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.18071)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SFT%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-SFT)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-RL%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-RL)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Eval%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-Benchmark)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/VanillaH1/AlignXplore-Plus)

</div>

## 1.  Introduction

<strong>AlignXplore+</strong> is a framework for transferable personalization in large language models. Building upon AlignXplore, it represents a substantial evolution of the approach by advancing a central idea: natural language can serve as a universal, model- and task-agnostic interface for expressing fine-grained and multi-dimensional user preferences.

## 2.  Key Features

- <strong> General-Purpose </strong>: AlignXplore+ operates in more realistic, real-world scenario, demonstrating that high-quality user preference summaries can be inferred from heterogeneous sources, including social networks, e-commerce platforms, and news streams.
- <strong> Transferable </strong>: Preference summaries inferred by AlignXplore+ demonstrate strong transferability across both tasks (e.g., from response selection to news recommendation) and models (e.g., from GPT-OSS-20B to Qwen2.5-7B-Instruct).
- <strong> Robust </strong>: Inherited from AlignXplore, AlignXplore+ supports preference reasoning from streaming inputs and maintains stable performance under noisy or imperfect signals, ensuring reliable personalization in realistic, dynamic settings.


<div align="center">
  <div style="display:inline-block; margin: 0 10px;">
    <img src="figures/Task_Transfer.PNG" width="45%"/><br/>
    <b>AlignX Framework</b>
  </div>

  <div style="display:inline-block; margin: 0 10px;">
    <img src="figures/Model_Transfer.PNG" width="45%"/><br/>
    <b>AlignXplore+ Framework</b>
  </div>
</div>

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
