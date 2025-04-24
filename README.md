
<div align="center">
<h1>
  360Zhinao3 (360智脑)
</h1>
</div>
<div align="center">
    🤗 <a href="https://huggingface.co/qihoo360">HuggingFace</a>&nbsp&nbsp | &nbsp&nbsp
    💬 <a href="./WeChat.png">WeChat (微信)</a>&nbsp&nbsp 
</div>
<br>
<p align="center">
 Feel free to visit 360Zhinao's official website<a href="https://ai.360.com"> https://ai.360.com</a> for more experience.
</p>

<br>

# Introduction
 🎉🎉🎉 **TRecently, Qihoo 360 has open sourced and upgraded its self-developed 7B parameter model 360Zhinao3-7B. It has now been launched on the Github open source community [360zhinao3](https://github.com/Qihoo360/360zhinao3) and can be used for commercial purposes free of charge. he capabilities of the model have been comprehensively improved. Compared with small parameter models with less than 10B, 360Zhinao3-7B has achieved excellent results of first place in multiple benchmarks.**

 - **360Zhinao3-7B**
 - **360Zhinao3-7B-Instruct**
 - **360Zhinao3-7B-O1.5**

Notable features of our 360Zhinao models are:

360Zhinao3-7B is continuously pre-trained with 700B high-quality tokens on the basis of 360Zhinao2-7B. The two models have exactly the same structure. The improvement in model performance mainly stems from the improvement in the quality of training data.


<br>

# News and Updates
- [2025.04.14] 🔥🔥🔥**We have released the 360Zhinao3 series of models, and at the same time opened up 360Zhinao3-7B, 360Zhinao3-7B-Instruct, and the long thought chain model 360Zhinao3-7B-O1.5.**
- [2024.11.18] We release 360Zhinao2-7B, providing access to both the Base model and Chat models with text lengths of 4K, 32K, and 360K.
- [2024.05.23] We released two models, 360Zhinao-search and 360Zhinao-1.8B-Reranking, which ranked first respectively in the Retrieval and Reranking tasks of [C-MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) .
- [2024.05.20] We extended llama3 and released **llama3-8B-360Zhinao-360k-Instruct**<a href="https://huggingface.co/qihoo360/llama3-8B-360Zhinao-360k-Instruct">🤗</a>
- [2024.04.12] We released **360Zhinao-7B** v1.0, including the base model and three chat models with context lengths 4K, 32K and 360K. 
Technical report is on [arXiv](https://arxiv.org/abs/2405.13386).

<br>

# Table of contents
- [Download URL](#Download-URL)
- [Model Evaluation](#Model-Evaluation)
- [Quickstart](#Quickstart)
- [Model Inference](#Model-Inference)
- [Model Finetune](#Model-Finetune)
- [License](#License)

<br>

# Download URL

| Size | Model | BF16 |
|:-:|-|:-:|
| 7B | 360Zhinao3-7B | <a href="https://huggingface.co/qihoo360/360Zhinao3-7B">🤗</a> |
| 7B | 360Zhinao3-7B-Instruct | <a href="https://huggingface.co/qihoo360/360Zhinao3-7B-Instruct">🤗</a> |
| 7B | 360Zhinao3-7B-O1.5 | <a href="https://huggingface.co/qihoo360/360Zhinao3-7B-O1.5">🤗</a> |

<br>

# Model Evaluation
## Base Model
We used the open source tool opencompass to conduct multi-dimensional evaluation of the model. The benchmark average score of the model ranks first among models with less than 10B parameters. It is competitive in the same size.

<table>
	<tr>
	    <td>Type</td><td>Datasets</td><td>language</td><td>glm4-9b</td><td>Qwen2.5-7B</td><td>internlm2.5-7b</td><td>Yi1.5-9B</td><td>gemma2-9b</td><td>Llama3.1-8B</td><td>360Zhinao2-7B</td><td>360Zhinao3-7B</td>
	</tr>
	<tr>
	    <td rowspan="5">Exam</td><td>ceval</td><td>zh</td><td>75.83</td><td>81.41</td><td>77.71</td><td>73.51</td><td>56.36</td><td>51.67</td><td>83.04</td><td><strong>84.7</strong></td>
	</tr>
    <tr>
        <td>mmlu</td><td>en</td><td>75.5</td><td>75.5</td><td>71.55</td><td>71.43</td><td>72.22</td><td>66.75</td><td>67.84</td><td>75.42</td>
    </tr>
    <tr>
        <td>cmmlu</td><td>zh</td><td>74.24</td><td>81.79</td><td>78.77</td><td>74.2</td><td>58.89</td><td>52.49</td><td>73.8</td><td><strong>82.17</strong></td>
    </tr>
    <tr>
        <td>ARC-c</td><td>en</td><td>94.92</td><td>80</td><td>85.08</td><td>87.46</td><td>77.63</td><td>80.68</td><td>87.12</td><td>88.14</td>
    </tr>
    <tr>
        <td>ARC-e</td><td>en</td><td>98.41</td><td>84.83</td><td>95.24</td><td>94.53</td><td>78.84</td><td>89.77</td><td>92.77</td><td>94</td>
    </tr>
    <tr>
        <td rowspan="2">Language</td><td>WiC</td><td>en</td><td>51.57</td><td>52.82</td><td>50.78</td><td>50.63</td><td>50.47</td><td>50</td><td>49.84</td><td>50.31</td>
    </tr>
    <tr>
        <td>WSC</td><td>en</td><td>68.27</td><td>68.27</td><td>69.23</td><td>66.35</td><td>68.27</td><td>67.31</td><td>65.38</td><td><strong>71.15</strong></td>
    </tr>
    <tr>
        <td rowspan="2">Knowledge</td>
        <td>BoolQ</td><td>en</td><td>81.8</td><td>83.88</td><td>89.51</td><td>84.46</td><td>85.6</td><td>82.2</td><td>88.29</td><td>88.38</td>
    </tr>
    <tr>
        <td>commonsense_qa</td><td>en</td><td>71.17</td><td>73.22</td><td>68.55</td><td>71.58</td><td>68.47</td><td>71.25</td><td>69.78</td><td>71.33</td>
    </tr>
    <tr>
        <td rowspan="6">Understanding</td>
        <td>C3</td><td>zh</td><td>91.51</td><td>92</td><td>93.04</td><td>85.86</td><td>81.64</td><td>83.51</td><td>93.26</td><td>92.77</td>
    </tr>
    <tr>
        <td>race-middle</td><td>en</td><td>91.99</td><td>91.02</td><td>92.06</td><td>91.16</td><td>88.09</td><td>81.69</td><td>90.46</td><td>90.04</td>
    </tr>
    <tr>
        <td>race-high</td><td>en</td><td>90.71</td><td>87.91</td><td>90.08</td><td>88.34</td><td>82.08</td><td>78.73</td><td>86.74</td><td>85.96</td>
    </tr>
    <tr>
        <td>lcsts</td><td>zh</td><td>18.29</td><td>15.82</td><td>15.96</td><td>16.49</td><td>10.62</td><td>17.29</td><td>18.61</td><td><strong>18.85</strong></td>
    </tr>
    <tr>
        <td>eprstmt-dev</td><td>zh</td><td>91.88</td><td>86.88</td><td>91.25</td><td>91.88</td><td>48.12</td><td>83.12</td><td>90</td><td><strong>92.50</strong></td>
    </tr>
    <tr>
        <td>lambada</td><td>en</td><td>71.67</td><td>71.14</td><td>69.98</td><td>70.64</td><td>75.43</td><td>74.23</td><td>72.56</td><td>68.17</td>
    </tr>
    <tr>
        <td rowspan="3">Reasoning</td>
        <td>hellaswag</td><td>en</td><td>70.25</td><td>72.76</td><td>70.38</td><td>71.55</td><td>66.83</td><td>74.65</td><td>71.49</td><td>73.61</td>
    </tr>
    <tr>
        <td>siqa</td><td>en</td><td>81.73</td><td>72.52</td><td>78.97</td><td>76.2</td><td>58.96</td><td>64.18</td><td>77.12</td><td>79.02</td>
    </tr>
    <tr>
        <td>bbh</td><td>en</td><td>73.68</td><td>54.63</td><td>59.43</td><td>67.86</td><td>68.45</td><td>59.9</td><td>46.54</td><td><strong>73.74</strong></td>
    </tr>
    <tr>
        <td rowspan="2">Code</td>
        <td>humaneval</td><td>en</td><td>69.51</td><td>75</td><td>60.37</td><td>26.22</td><td>5.49</td><td>27.44</td><td>60.98</td><td>64.63</td>
    </tr>
    <tr>
        <td>mbpp</td><td>en</td><td>60</td><td>60</td><td>43.6</td><td>56.8</td><td>51.2</td><td>42.6</td><td>54</td><td><strong>67.80</strong></td>
    </tr>
    <tr>
        <td rowspan="2">Math</td>
        <td>math</td><td>en</td><td>26.86</td><td>38</td><td>27.14</td><td>27.06</td><td>28.52</td><td>15.32</td><td>38.34</td><td>37.60</td>
    </tr>
    <tr>
        <td>gsm8k</td><td>en</td><td>78.54</td><td>79.76</td><td>52.54</td><td>71.11</td><td>73.09</td><td>56.25</td><td>75.51</td><td>78.77</td>
    </tr>
    <tr>
        <td rowspan="2">Overall</td>
        <td>avg_zh</td><td></td><td>70.35</td><td>71.58</td><td>71.35</td><td>68.39</td><td>51.13</td><td>57.62</td><td>71.74</td><td><strong>74.20</strong></td>
    </tr>
    <tr>
        <td>avg_all</td><td></td><td>73.11</td><td>71.78</td><td>69.60</td><td>68.88</td><td>61.60</td><td>62.32</td><td>70.61</td><td><strong>74.83</strong></td>
    </tr>
</table>


<br>


## Instruct Model

We have evaluated and compared the 360Zhinao3-7B-Instruct model on three popular evaluations: IFEval, MT-bench, and CF-Bench. MT-bench and CFBench both rank first among open-source models of the same level and have strong competitiveness. In IFEval (prompt strict), it is second only to glm4-9b and has the highest score in the 7B size.

| Model                 | MT-bench | IFEval(strict prompt) | CFBench(CSR,ISR,PSR) |          |          |
|-----------------------|----------|-----------------------|----------------------|----------|----------|
| Qwen2.5-7B-Instruct   | 8.07     | 0.556                 | 0.81                 | 0.46     | 0.57     |
| Yi-9B-16k-Chat        | 7.44     | 0.455                 | 0.75                 | 0.4      | 0.52     |
| GLM4-9B-Chat          | 8.08     | **0.634**             | 0.82                 | 0.48     | 0.61     |
| InternLM2.5-7B-Chat   | 7.39     | 0.540                 | 0.78                 | 0.4      | 0.54     |
| 360Zhinao2-7B-Chat-4k | 7.86     | 0.577                 | 0.8                  | 0.44     | 0.57     |
| 360Zhinao3-7B-Instruct| **8.17** | 0.626                 | **0.83**             | **0.52** | **0.64** |

## Long COT Model
We used the previously open-sourced [Light-R1](https://github.com/Qihoo360/Light-R1) method of Zhinao to continue fine-tuning the Long COT of 360Zhinao3-7B-Instruct, as well as RFT and GRPO. There is still a certain gap compared with the latest OpenThinker2-7B, but it surpasses all previous models based on the general Qwen2.5-7B-Instruct.

| Model | Date | Base Model | AIME24 | AIME25 | GPQA Diamond |
| ---- | ---- | ---- | ---- | ---- | ---- |
| OpenThinker2-7B | 25.4.3 | Qwen2.5-7B-Instruct | 50 | 33.3 | 49.3 |
| OpenThinker-7B | 25.1.28 | Qwen2.5-7B-Instruct | 31.3 | 23.3 | 42.4 |
| 360Zhinao3-7B-O1.5 | 25.4.14 | 360Zhinao3-7B-Instruct | 54.2 | 36.3 | 40.0 |
| OpenR1-Qwen-7B | 25.2.11 | Qwen2.5-Math-7B-Instruct | 48.7 | 34.7 | 21.2 |
| DeepSeek-R1-Distill-Qwen-7B | 25.1.20 | Qwen2.5-Math-7B-Instruct | 57.3 | 33.3 | 47.3 |
| Light-R1-7B-DS | 25.3.12 | DeepSeek-R1-Distill-Qwen-7B | 59.1 | 44.3 | 49.4 |
| Areal-boba-RL-7B | 25.3.31 | DeepSeek-R1-Distill-Qwen-7B | 61.9 | 48.3 | 47.6 |


<br>

# Quickstart
A simple example to illustrate how to quickly use 360Zhinao3-7B, 360Zhinao3-7B-Instruct, and 360Zhinao3-7B-O1.5 with 🤗Transformers

## 🤗 Transformers
### Demonstration of Base Model Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao3-7B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, 
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True).cuda()

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)
generation_config.max_new_tokens = 1024

inputs = tokenizer('中国二十四节气\n1. 立春\n2. 雨水\n3. 惊蛰\n4. 春分\n5. 清明\n', return_tensors='pt')
inputs = inputs.to(model.device)

pred = model.generate(input_ids=inputs["input_ids"], generation_config=generation_config)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

# Demonstration of Instruct Model Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao3-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True).cuda()

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)
generation_config.max_new_tokens = 2048

messages = []

#round-1
print(f"user: 简单介绍一下刘德华")
messages.append({"role": "user", "content": "简单介绍一下刘德华"})
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
pred = model.generate(input_ids=input_ids, generation_config=generation_config)
response = tokenizer.decode(pred.cpu()[0][len(input_ids[0]):], skip_special_tokens=True)
messages.append({"role": "assistant", "content": response})
print(f"gpt: {response}")


#round-1
print(f"user: 他有什么代表作?")
messages.append({"role": "user", "content": "他有什么代表作?"})
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
pred = model.generate(input_ids=input_ids, generation_config=generation_config)
response = tokenizer.decode(pred.cpu()[0][len(input_ids[0]):], skip_special_tokens=True)
messages.append({"role": "assistant", "content": response})
print(f"gpt: {response}")
```
# Demonstration of Long COT Model Inference
```python
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao3-7B-O1.5"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True).cuda()

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)
generation_config.max_new_tokens = 2048


def extract_thinking_and_answer(input_string):
    thinking, answer = "", ""
    # 提取答案
    pattern_answer = r'.*</think>(.*)$'
    match_answer = re.search(pattern_answer, input_string, re.S)
    if match_answer:
        answer = match_answer.group(1)
    else:
        return thinking, input_string

    # 提取思考过程
    pattern_thinking = r'<think>(.*?)</think>'
    match_thinking = re.search(pattern_thinking, input_string, re.S)
    if match_thinking:
        thinking = match_thinking.group(1)

    return thinking, answer


messages = []
messages.append({"role": "user", "content": "现有一笼子，里面有鸡和兔子若干只，数一数，共有头14个，腿38条，求鸡和兔子各有多少只？"})
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
pred = model.generate(input_ids=input_ids, generation_config=generation_config)
response = tokenizer.decode(pred.cpu()[0][len(input_ids[0]):], skip_special_tokens=True)
thinking, answer = extract_thinking_and_answer(response)
messages.append({"role": "assistant", "content": answer, "reasoning_content": thinking})
print(json.dumps(messages, ensure_ascii=False, indent=4))
```

<br>

# Model Inference
## Deployment
### vLLM Installation
We recommend using  `vllm==0.6.0`.

If you are using **CUDA 12.1 and PyTorch 2.1**, you can install vLLM directly with:
```shell
pip install  vllm==0.6.0
```

Otherwise, please refer to the official vLLM [Installation Instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html).

After installation, perform the following steps:
1. Copy `vllm/zhinao.py` into `vllm/model_executor/models` in your vllm installation directory (in python/conda env).
2. Then add a line in `vllm/model_executor/models/__init__.py`

    ```shell
    "ZhinaoForCausalLM": ("zhinao", "ZhinaoForCausalLM"),
    ```

### vLLM Service Start

Start the service:
```shell
python -m vllm.entrypoints.openai.api_server \
    --model qihoo360/360Zhinao3-7B-O1.5 \
    --served-model-name 360Zhinao3-7B-O1.5 \
    --port 8360 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code
```

Use curl to request the service:
```shell
curl http://localhost:8360/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "360Zhinao3-7B-O1.5",
    "max_tokens": 200,
    "top_k": -1,
    "top_p": 0.8,
    "temperature": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ],
    "stop": [
        "<eod>",
        "<|im_end|>",
        "<|im_start|>"
    ]
}'

```

Use python to request the service:
```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8360/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="360Zhinao3-7B-O1.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ],
    stop=[
        "<eod>",
        "<|im_end|>",
        "<|im_start|>"
    ],
    presence_penalty=0.0,
    frequency_penalty=0.0
)
print("Chat response:", chat_response)
```

> If you need to enable repetition penalty, we recommend setting `presence_penalty` and `frequency_penalty` instead of `repetition_penalty`.


<br>

# Model Finetune
## Training data

Training Data: `data/training_data_sample.json`. This example data has 10,000 rows sampled from [multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) with converted format.

Data Format:
```json
[
  {
    "id": 1,
    "conversations": [
        {
            "from": "system",
            "value": "You are a helpful assistant."
        },
        {
            "from": "user",
            "value": "您好啊"
        },
        {
            "from": "assistant",
            "value": "你好！我今天能为您做些什么？有什么问题或需要帮助吗? 我在这里为您提供服务。"
        }
    ]
  }
]
```
## Finetuning scripts
```shell
set -x

HOSTFILE=hostfile
DS_CONFIG=./finetune/ds_config_zero2.json

# PARAMS
LR=5e-6
EPOCHS=3
MAX_LEN=32768
BATCH_SIZE=4
NUM_NODES=1
NUM_GPUS=8
MASTER_PORT=29500

IS_CONCAT=False # Whether to concatenate to maximum length (MAX_LEN)

DATA_PATH="./data/training_data_sample.json"
MODEL_PATH="qihoo360/360Zhinao3-7B-Instruct"
OUTPUT_DIR="./outputs/"

deepspeed --hostfile ${HOSTFILE} \
        --master_port ${MASTER_PORT} \
        --num_nodes ${NUM_NODES} \
        --num_gpus ${NUM_GPUS} \
        finetune.py \
        --report_to "tensorboard" \
        --data_path ${DATA_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --model_max_length ${MAX_LEN} \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --save_strategy steps \
        --save_steps 200 \
        --learning_rate ${LR} \
        --lr_scheduler_type cosine \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --weight_decay 0.1 \
        --warmup_ratio 0.01 \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --deepspeed ${DS_CONFIG} \
        --is_concat ${IS_CONCAT} \
        --logging_steps 1 \
        --log_on_each_node False
```
```shell
bash finetune/ds_finetune.sh
```
- Configuring `HOSTFILE` switches between single-machine and multi-machine training.
- configuring `ds_config` switches between zero1, zero2 and zero3.
- `fp16, bf16` could configure mixed precision training. bf16 is recommended to be consistent with the pretrained model.
- `is_concat` configures whether the training data is concatenated or not.

<br>

# License

The source code of this repository follows the open-source license Apache 2.0.

360​Zhinao open-source models support free commercial use. It is not necessary for you to submit a request for commercial usage. 
