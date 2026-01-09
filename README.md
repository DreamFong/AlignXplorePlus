

<div align="center">
  <h1 style="font-size: 40px;">AlignXplore+</h1>
  <p>Text as a Universal Interface for Personalizing Large Language Models</p>

  
  [![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2601.04963)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SFT%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-SFT)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-RL%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-RL)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Eval%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-Benchmark)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/VanillaH1/AlignXplore-Plus)

</div>

## 📖 Introduction

<strong>AlignXplore+</strong> is a framework for transferable personalization in large language models. Building upon AlignXplore, it represents a substantial evolution of the approach by advancing a central idea: natural language can serve as a universal, model- and task-agnostic interface for expressing fine-grained and multi-dimensional user preferences.

![Model](./figures/model.jpeg)

## ✨ Key Features

- <strong> General-Purpose </strong>: AlignXplore+ operates in a more realistic, real-world scenario, demonstrating that high-quality user preference summaries can be inferred from heterogeneous sources, including social networks, e-commerce platforms, and news streams.
- <strong> Transferable </strong>: Preference summaries inferred by AlignXplore+ demonstrate strong transferability across both tasks (e.g., from response selection to news recommendation) and models (e.g., from GPT-OSS-20B to Qwen2.5-7B-Instruct).
- <strong> Streaming & Robust </strong>: Inherited from AlignXplore, AlignXplore+ supports preference reasoning from streaming inputs and maintains stable performance under noisy or imperfect signals, ensuring reliable personalization in realistic, dynamic settings.

<!-- <div align="center">
<table border="0" cellpadding="0" cellspacing="0">
  <tr>
    <td align="center" valign="top">
      <img src="figures/Model_Transfer.PNG" width="420"><br>
      <small><b>Figure 1.</b> Cross-model transferability of preference summaries</small>
    </td>
   <td align="center" valign="top">
      <img src="figures/Task_Transfer.PNG" width="420"><br>
     <small><b>Figure 2.</b> Cross-domain transfer performance</small>
    </td>
  </tr>
</table>
</div> -->

## 📊 Results

### Main Results

The following is our main results across nine benchmarks. P-Soups are split into three preference dimensions: expertise, informativeness, and style. We compare models in three settings: **direct sequence modeling**, **full-history preference inference**, and **streaming preference inference**. All preference inference models (both full-history and streaming) use the Qwen3-8B-non-thinking as the downstream model. Within each setting, **Bold** and <u>Underline</u> mark the best and second-best results among models at the $\sim$ 8B scale. <span style="color:gray;">Gray</span> score highlights models that are outperformed by the best-performing $\sim$ 8B model in the same column, including direct sequence models and larger preference inference models that do not show a performance advantage.

<table style="border-collapse:collapse; width:100%; text-align:center;">
  <thead>
    <tr>
      <th rowspan="3" style="border:1px solid #ddd; padding:4px;">Model</th>
      <th colspan="3" style="border:1px solid #ddd; padding:4px;">In-domain</th>
      <th colspan="6" style="border:1px solid #ddd; padding:4px;">Out-of-domain</th>
      <th rowspan="3" style="border:1px solid #ddd; padding:4px;">Avg.</th>
    </tr>
    <tr>
      <th style="border:1px solid #ddd; padding:4px;">MIND</th>
      <th style="border:1px solid #ddd; padding:4px;">Amazon</th>
      <th style="border:1px solid #ddd; padding:4px;">AlignX</th>
      <th style="border:1px solid #ddd; padding:4px;">MovieLens</th>
      <th style="border:1px solid #ddd; padding:4px;">PersonaMem</th>
      <th style="border:1px solid #ddd; padding:4px;">Info.</th>
      <th style="border:1px solid #ddd; padding:4px;">Style</th>
      <th style="border:1px solid #ddd; padding:4px;">Expertise</th>
      <th style="border:1px solid #ddd; padding:4px;">HiCUPID</th>
    </tr>
    <tr>
      <th style="border:1px solid #ddd; padding:4px;">(Rec.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(Rec.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(Rec.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.G.)</th>
    </tr>
  </thead>
  <tbody>
    <!-- Direct Full-history Sequence Models w/o Preference Inference -->
    <tr>
      <td colspan="11" style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold; background:#f7f7f7;">
        Direct Full-history Sequence Models w/o Preference Inference
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-8B<sub>non-thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">63.03</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">84.05</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">59.63</span></td>
      <td style="border:1px solid #ddd; padding:4px;">88.57</td>
      <td style="border:1px solid #ddd; padding:4px;">61.40</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">46.84</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">42.33</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">38.33</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">47.02</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">59.02</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">TALLRec</td>
      <td style="border:1px solid #ddd; padding:4px;">81.96</td>
      <td style="border:1px solid #ddd; padding:4px;">94.91</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">66.30</span></td>
      <td style="border:1px solid #ddd; padding:4px;">97.90</td>
      <td style="border:1px solid #ddd; padding:4px;">64.36</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">51.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">70.16</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">60.16</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">47.41</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">70.53</span></td>
    </tr>
    <!-- Full-history Preference Inference -->
    <tr>
      <td colspan="11" style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold; background:#f7f7f7;">
        Full-history Preference Inference
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DeepSeek-R1-671B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">65.53</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">82.15</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">65.90</span></td>
      <td style="border:1px solid #ddd; padding:4px;">82.76</td>
      <td style="border:1px solid #ddd; padding:4px;">61.44</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">72.59</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">85.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">82.33</span></td>
      <td style="border:1px solid #ddd; padding:4px;">63.90</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.58</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-32B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">67.63</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">85.69</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">75.43</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">57.36</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.25</span></td>
      <td style="border:1px solid #ddd; padding:4px;">88.00</td>
      <td style="border:1px solid #ddd; padding:4px;">83.66</td>
      <td style="border:1px solid #ddd; padding:4px;">63.44</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.26</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">GPT-OSS-20B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.16</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">83.75</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">55.63</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">74.46</span></td>
      <td style="border:1px solid #ddd; padding:4px;">61.74</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">68.77</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">86.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">81.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">62.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">70.90</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-8B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>66.10</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>84.68</u></td>
      <td style="border:1px solid #ddd; padding:4px;">62.73</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>75.13</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>54.36</u></td>
      <td style="border:1px solid #ddd; padding:4px;">75.08</td>
      <td style="border:1px solid #ddd; padding:4px;"><b>87.50</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>83.50</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>60.05</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>72.12</u></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DS-R1-Distill-Qwen-7B</td>
      <td style="border:1px solid #ddd; padding:4px;">61.20</td>
      <td style="border:1px solid #ddd; padding:4px;">82.82</td>
      <td style="border:1px solid #ddd; padding:4px;">54.03</td>
      <td style="border:1px solid #ddd; padding:4px;">70.30</td>
      <td style="border:1px solid #ddd; padding:4px;">49.28</td>
      <td style="border:1px solid #ddd; padding:4px;">56.14</td>
      <td style="border:1px solid #ddd; padding:4px;">65.83</td>
      <td style="border:1px solid #ddd; padding:4px;">66.00</td>
      <td style="border:1px solid #ddd; padding:4px;">60.01</td>
      <td style="border:1px solid #ddd; padding:4px;">62.84</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore</span></td>
      <td style="border:1px solid #ddd; padding:4px;">61.23</td>
      <td style="border:1px solid #ddd; padding:4px;">78.58</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>66.60</u></td>
      <td style="border:1px solid #ddd; padding:4px;">69.93</td>
      <td style="border:1px solid #ddd; padding:4px;">53.98</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>76.24</u></td>
      <td style="border:1px solid #ddd; padding:4px;">78.00</td>
      <td style="border:1px solid #ddd; padding:4px;">72.66</td>
      <td style="border:1px solid #ddd; padding:4px;">53.50</td>
      <td style="border:1px solid #ddd; padding:4px;">66.07</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore+</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>71.36</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>86.39</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>75.03</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>75.80</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>58.08</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>78.07</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>86.33</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>82.50</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>62.42</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>75.10</b></td>
    </tr>
    <!-- Streaming Preference Inference -->
    <tr>
      <td colspan="11" style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold; background:#f7f7f7;">
        Streaming Preference Inference
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DeepSeek-R1-671B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.30</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">80.54</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.06</span></td>
      <td style="border:1px solid #ddd; padding:4px;">83.63</td>
      <td style="border:1px solid #ddd; padding:4px;">58.96</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">66.61</span></td>
      <td style="border:1px solid #ddd; padding:4px;">85.00</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">79.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">60.32</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">72.93</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-32B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">66.60</span></td>
      <td style="border:1px solid #ddd; padding:4px;">85.35</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.60</span></td>
      <td style="border:1px solid #ddd; padding:4px;">77.78</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">53.26</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.58</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">83.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">81.67</span></td>
      <td style="border:1px solid #ddd; padding:4px;">59.83</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">71.81</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">GPT-OSS-20B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">84.55</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">56.86</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;">54.82</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">69.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">83.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">77.50</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">59.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">69.46</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-8B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>66.13</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>83.58</u></td>
      <td style="border:1px solid #ddd; padding:4px;">62.90</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>75.97</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>51.68</u></td>
      <td style="border:1px solid #ddd; padding:4px;">74.08</td>
      <td style="border:1px solid #ddd; padding:4px;"><b>85.00</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>82.66</b></td>
      <td style="border:1px solid #ddd; padding:4px;">59.17</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>71.24</u></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DS-R1-Distill-Qwen-7B</td>
      <td style="border:1px solid #ddd; padding:4px;">61.16</td>
      <td style="border:1px solid #ddd; padding:4px;">81.78</td>
      <td style="border:1px solid #ddd; padding:4px;">56.40</td>
      <td style="border:1px solid #ddd; padding:4px;">69.63</td>
      <td style="border:1px solid #ddd; padding:4px;">46.64</td>
      <td style="border:1px solid #ddd; padding:4px;">58.63</td>
      <td style="border:1px solid #ddd; padding:4px;">60.83</td>
      <td style="border:1px solid #ddd; padding:4px;">64.16</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>59.29</u></td>
      <td style="border:1px solid #ddd; padding:4px;">62.05</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore</span></td>
      <td style="border:1px solid #ddd; padding:4px;">60.66</td>
      <td style="border:1px solid #ddd; padding:4px;">79.01</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>69.90</u></td>
      <td style="border:1px solid #ddd; padding:4px;">67.96</td>
      <td style="border:1px solid #ddd; padding:4px;">48.42</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>74.41</u></td>
      <td style="border:1px solid #ddd; padding:4px;">74.83</td>
      <td style="border:1px solid #ddd; padding:4px;">69.16</td>
      <td style="border:1px solid #ddd; padding:4px;">50.34</td>
      <td style="border:1px solid #ddd; padding:4px;">66.07</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore+</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>71.80</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>85.35</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>73.67</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>77.23</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>54.58</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>76.57</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>80.33</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>78.50</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>60.51</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>73.17</b></td>
    </tr>
  </tbody>
</table>


### Cross-model Transfer Results


### Cross-task Transfer Results


## 🚀 Quick Start

### Requirments

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

### SFT Training

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

### RL Training 

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

### Inference

```bash
cd eval

# You should modify line #21 to the path of your model.
bash gen_preference.sh
bash straming_gen_preference.sh
# You should modify line #21 to the model name of your model.
bash evaluation.sh
```

## ✒️ Citation

```bibtex
@article{liu2026text,
  title={Text as a Universal Interface for Transferable Personalization},
  author={Yuting Liu, Jian Guan, Jia-Nan Li, Wei Wu, Jiang-Ming Yang, Jianzhe Zhao, Guibing Guo},
  journal={arXiv preprint arXiv:2601.04963},
  year={2026}
```
