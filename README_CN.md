<div align="center">
<h1>
  360Zhinao3 (360智脑)
</h1>
</div>
<div align="center">
    🤗 <a href="https://huggingface.co/qihoo360">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp
    💬 <a href="./WeChat.png">WeChat (微信)</a>&nbsp&nbsp
</div>
<br>
<p align="center">
 欢迎访问360智脑官网<a href="https://ai.360.com"> https://ai.360.com </a>体验更多更强大的功能。
</p>

<br>

# 模型介绍
 🎉🎉🎉**近日，奇虎360开源升级了自研的7B参数模型360Zhinao3-7B，现已上线Github开源社区 [**360zhinao3**](https://github.com/Qihoo360/360zhinao3)，可免费商用。模型各项能力得到全面提升，与10B以下的小参数量模型进行对比，360Zhinao3-7B在多个benchmark上，均取得第一的优异成绩。**

开源模型包括：
 - **360Zhinao3-7B**
 - **360Zhinao3-7B-Instruct**
 - **360Zhinao3-7B-O1.5**

360Zhinao3-7B 是在 360Zhinao2-7B 的基础上继续预训练了700B的高质量token，两者模型结构完全相同。模型效果上的提升，主要源于训练数据的质量的提升。

<br>

# 更新信息
- [2025.04.14] 🔥🔥🔥**我们发布了360Zhinao3系列模型，同时开放 360Zhinao3-7B、360Zhinao3-7B-Instruct 以及长思维链模型 360Zhinao3-7B-O1.5。**
- [2024.11.18] 我们发布了360Zhinao2-7B，同时开放Base模型和4K、32K、360K三种文本长度的Chat模型。
- [2024.05.23] 我们发布了360Zhinao-search以及360Zhinao-1.8B-Reranking两个模型，分别在[C-MTEB 榜单](https://huggingface.co/spaces/mteb/leaderboard)的Retrieval和Reranking任务上排名第一。
- [2024.05.20] 我们将llama3的窗口长度扩展到360k并发布了**llama3-8B-360Zhinao-360k-Instruct**<a href="https://huggingface.co/qihoo360/llama3-8B-360Zhinao-360k-Instruct">🤗</a>
- [2024.04.12] 我们发布了360Zhinao-7B 1.0版本，同时开放Base模型和4K、32K、360K三种文本长度的Chat模型。
技术报告详见[arXiv](https://arxiv.org/abs/2405.13386)。

<br>

# 目录
- [下载地址](#下载地址)
- [模型评估](#模型评估)
- [快速开始](#快速开始)
- [模型推理](#模型推理)
- [模型微调](#模型微调)
- [许可证](#许可证)

<br>

# 下载地址
本次发布版本和下载链接见下表：
| Size | Model | BF16 |
|:-:|-|:-:|
| 7B | 360Zhinao3-7B | <a href="https://huggingface.co/qihoo360/360Zhinao3-7B">🤗</a> |
| 7B | 360Zhinao3-7B-Instruct | <a href="https://huggingface.co/qihoo360/360Zhinao3-7B-Instruct">🤗</a> |
| 7B | 360Zhinao3-7B-O1.5 | <a href="https://huggingface.co/qihoo360/360Zhinao3-7B-O1.5">🤗</a> |

<br>

# 模型评估

## 基础模型效果

我们使用了开源工具opencompass对模型进行多维度评估，模型的benchmark平均分在10B以下模型中排名第一，同尺寸具备竞争力。

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


## Instruct模型效果

我们在IFEval、MT-bench、CF-Bench三个流行的评测上对360Zhinao3-7B-Instruct模型进行了评测比较，MT-bench 和CFBench均在同级别开源模型中排名第一，具备较强竞争力。在IFEval (prompt strict) 仅次于glm4-9b，在7B尺寸上得分最高:

| Model                 | MT-bench | IFEval(strict prompt) | CFBench(CSR,ISR,PSR) |          |          |
|-----------------------|----------|-----------------------|----------------------|----------|----------|
| Qwen2.5-7B-Instruct   | 8.07     | 0.556                 | 0.81                 | 0.46     | 0.57     |
| Yi-9B-16k-Chat        | 7.44     | 0.455                 | 0.75                 | 0.4      | 0.52     |
| GLM4-9B-Chat          | 8.08     | **0.634**             | 0.82                 | 0.48     | 0.61     |
| InternLM2.5-7B-Chat   | 7.39     | 0.540                 | 0.78                 | 0.4      | 0.54     |
| 360Zhinao2-7B-Chat-4k | 7.86     | 0.577                 | 0.8                  | 0.44     | 0.57     |
| 360Zhinao3-7B-Instruct| **8.17** | 0.626                 | **0.83**             | **0.52** | **0.64** |

## 长思维链模型效果
我们用之前智脑开源的 [Light-R1](https://github.com/Qihoo360/Light-R1) 方法对360Zhinao3-7B-Instruct进行了长思维链的继续微调和RFT，GRPO。与最新的OpenThinker2-7B有一定差距，超越了之前的所有以通用Qwen2.5-7B-Instruct为基座的模型。

| Model | Date | Base Model | AIME24 | AIME25 | GPQA Diamond |
| ---- | ---- | ---- | ---- | ---- | ---- |
| OpenThinker2-7B | 25.4.3 | Qwen2.5-7B-Instruct | 50 | 33.3 | 49.3 |
| OpenThinker-7B | 25.1.28 | Qwen2.5-7B-Instruct | 31.3 | 23.3 | 42.4 |
| 360Zhinao3-7B-O1.5 | 25.4.14 | 360Zhinao3-7B-Instruct | 54.2 | 36.3 | 40.0 |
| OpenR1-Qwen-7B | 25.2.11 | Qwen2.5-Math-7B-Instruct | 48.7 | 34.7 | 21.2 |
| DeepSeek-R1-Distill-Qwen-7B | 25.1.20 | Qwen2.5-Math-7B-Instruct | 57.3 | 33.3 | 47.3 |
| Light-R1-7B-DS | 25.3.12 | DeepSeek-R1-Distill-Qwen-7B | 59.1 | 44.3 | 49.4 |
| Areal-boba-RL-7B | 25.3.31 | DeepSeek-R1-Distill-Qwen-7B | 61.9 | 48.3 | 47.6 |


# 快速开始
简单的示例来说明如何利用 🤗Transformers 快速使用 360Zhinao3-7B、360Zhinao3-7B-Instruct 以及 360Zhinao3-7B-O1.5

## 🤗Transformers
### Base模型推理

此代码演示使用transformers快速使用360Zhinao2-7B-Base模型进行推理
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

### Instruct模型推理

此代码演示使用 🤗Transformers 快速使用 360Zhinao3-7B-Instruct 模型进行推理
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

### 长思维链模型推理
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

# 模型推理
## 模型部署
### vLLM安装环境
如希望部署及加速推理，我们建议你使用 `vllm==0.6.0`。

如果你使用**CUDA 12.1和PyTorch 2.1**，可以直接使用以下命令安装vLLM。
```shell
pip install vllm==0.6.0
```

否则请参考vLLM官方的[安装说明](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

>安装完成后，还需要以下操作~
1. 把vllm/zhinao.py文件复制到env环境对应的vllm/model_executor/models目录下。
2. 然后在vllm/model_executor/models/\_\_init\_\_.py文件增加一行代码

    ```shell
    "ZhinaoForCausalLM": ("zhinao", "ZhinaoForCausalLM"),
    ```

### vLLM服务启动

启动服务
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

使用curl请求服务
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

使用python请求服务
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

> 注意：如需要开启重复惩罚，建议使用 *presence_penalty* 和 *frequency_penalty* 参数。

<br>

# 模型微调
## 训练数据

我们提供了微调训练样例数据 data/test.json，该样例数据是从 [multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) 采样出 1 万条，并且做了格式转换。

数据格式:
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

## 微调训练
训练脚本如下：
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

IS_CONCAT=False # 是否数据拼接到最大长度（MAX_LEN）

DATA_PATH="./data/training_data_sample.json"
MODEL_PATH="qihoo360/360Zhinao3-7B"
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
- 可通过配置hostfile，实现单机、多机训练。
- 可通过配置ds_config，实现zero2、zero3。
- 可通过配置fp16、bf16实现混合精度训练，建议使用bf16，与预训练模型保持一致。
- 可通过配置is_concat参数，控制训练数据是否拼接，当训练数据量级较大时，可通过拼接提升训练效率。

<br>

# 许可证

本仓库源码遵循开源许可证Apache 2.0。

360智脑开源模型支持免费商用，无需向我们进行特殊申请。
