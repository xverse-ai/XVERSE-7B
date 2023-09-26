<div align="center">
<h1>
  XVERSE-7B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-7B">🤗 XVERSE-7B</a>&nbsp｜&nbsp<a href="https://huggingface.co/xverse/XVERSE-7B-Chat">🤗 XVERSE-7B-Chat</a>&nbsp｜&nbsp
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜&nbsp
        <a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <a href="README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

## Model Introduction

**XVERSE-7B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology. The models released this time include the base model **XVERSE-7B** and the chat model **XVERSE-7B-Chat**. Its key features are as follows:

- **Model Structure**: XVERSE-7B uses the mainstream Decoder-only Transformer network structure, supports 8k context length, which can meet the need of longer multi-round dialogues, knowledge question-answering, and summarization. This makes the model more versatile in application scenarios.
- **Training Data**: The model has been thoroughly trained on a diversified and high-quality dataset consisting of 2.6 trillion of tokens, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages.
- **Tokenization**: Based on the BPE (Byte-Pair Encoding) algorithm, a tokenizer with a vocabulary size of 100,534 has been trained using hundreds of gigabytes of language data. This tokenizer is capable of supporting multilingual without the need for additional vocabulary expansion.
- **Training Framework**: Several key technologies have also been independently developed, including efficient operators, memory optimization, parallel scheduling strategies, overlap of data-computation-communication, and synergy between platforms and frameworks. These advancements enhance training efficiency and model stability. With these technologies, the peak computational power utilization rate on a thousand-card cluster can reach 58.5%, ranking at the forefront of the industry.

## Model Evaluation

In order to validate the various abilities of the model, we have chosen several comprehensive capability benchmarks across multiple disciplines, including [MMLU](https://arxiv.org/abs/2009.03300) (English), [C-Eval](https://cevalbenchmark.com/) (Chinese), [AGIEval](https://arxiv.org/abs/2304.06364) (Chinese and English), [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) (Chinese and English), [GAOKAO-English](https://github.com/ExpressAI/AI-Gaokao) (English), the evaluation results are as follows (the bolded score represent the best performances):
|       Models       |    Type    |       MMLU       |      C-Eval      | AGIEval<sup>1</sup> | GAOKAO-Bench<sup>1</sup> | GAOKAO-English<sup>1</sup> |
| :----------------: | :--------: | :--------------: | :--------------: | :-----------------: | :----------------------: | :------------------------: |
|    Baichuan-7B     | pretrained | 42.3<sup>2</sup> | 42.8<sup>2</sup> |  34.4<sup>2</sup>   |     36.3<sup>2</sup>     |            44.3            |
| Baichuan2-7B-Base  | pretrained | 54.2<sup>2</sup> | 54.0<sup>2</sup> |  42.7<sup>2</sup>   |     47.5<sup>2</sup>     |            53.1            |
| Baichuan2-7B-Chat  | fine-tuned |       53.2       |       52.2       |        41.3         |           49.7           |            66.6            |
|    ChatGLM2-6B     | fine-tuned | 45.5<sup>2</sup> | 50.1<sup>2</sup> |        42.6         |           54.2           |            59.7            |
|     Falcon-7B      | pretrained | 27.8<sup>2</sup> |       25.8       |        26.2         |           26.3           |            29.9            |
|    InternLM-7B     | pretrained | 51.0<sup>2</sup> |       52.4       |        34.1         |           53.6           |            32.3            |
|  InternLM-7B-Chat  | fine-tuned | 50.8<sup>2</sup> |       52.8       |        39.0         |         **67.4**         |            43.9            |
|      Llama-7B      | pretrained | 35.1<sup>2</sup> |       27.0       |        27.4         |           26.0           |            30.1            |
|     Llama-2-7B     | pretrained | 45.3<sup>2</sup> |       28.9       |        27.0         |           27.8           |            47.8            |
|       MPT-7B       | pretrained | 29.6<sup>2</sup> |       27.8       |        24.2         |           25.3           |            28.1            |
|   Vicuna-7B-v1.5   | fine-tuned | 49.8<sup>2</sup> |       22.9       |        26.7         |           24.4           |            61.1            |
|   **XVERSE-7B**    | pretrained |       56.6       |     **57.1**     |        46.9         |           61.7           |            71.1            |
| **XVERSE-7B-Chat** | fine-tuned |     **63.7**     |       55.4       |      **48.9**       |           57.5           |          **78.2**          |

> <sup>1: Tests are conducted only on single-answer multiple-choice questions, thus excluding fill-in-the-blanks, open-ended questions, and multiple-answer multiple-choice questions.</sup>   
> <sup>2: Reporting results from official results of each model.</sup>   
>
> For MMLU, we adopt the [evaluation tools](https://github.com/hendrycks/test) provided by the authors, C-Eval, AGIEval, GAOKAO-Bench, GAOKAO-English are the same as MMLU, and uniformly use **5-shot** to construct the test samples.

### MMLU Category Results
|       Models       |    Type    | Average  |   STEM   | Social Science | Humanities |  Others  |
| :----------------: | :--------: | :------: | :------: | :------------: | :--------: | :------: |
|    Baichuan-7B     | pretrained |   42.3   |   35.6   |      48.9      |    38.4    |   48.1   |
| Baichuan2-7B-Chat  | fine-tuned |   53.2   |   43.1   |      59.1      |    50.0    |   59.1   |
|    ChatGLM2-6B     | fine-tuned |   45.5   |   40.1   |      51.6      |    41.2    |   51.2   |
|    InternLM-7B     | pretrained |   51.0   | **58.7** |      43.5      |    52.7    |   53.2   |
|      LLaMA-7B      | pretrained |   35.1   |   30.5   |      38.3      |    34.0    |   38.1   |
|     LLaMA2-7B      | pretrained |   45.3   |   36.4   |      51.2      |    42.9    |   52.2   |
|   **XVERSE-7B**    | pretrained |   56.6   |   45.6   |      65.3      |    50.4    |   65.5   |
| **XVERSE-7B-Chat** | fine-tuned | **63.7** |   51.7   |    **72.5**    |  **58.2**  | **72.2** |

### C-Eval Category Results
|       Models       |    Type    | Average  |   STEM   | Social Science | Humanities |  Others  |
| :----------------: | :--------: | :------: | :------: | :------------: | :--------: | :------: |
|    Baichuan-7B     | pretrained |   42.8   |   38.2   |      52.0      |    46.2    |   39.3   |
| Baichuan2-7B-Base  | pretrained |   54.9   |   47.9   |      67.3      |    58.4    |   52.8   |
| Baichuan2-7B-Chat  | fine-tuned |   52.2   |   44.6   |      65.0      |    55.8    |   50.9   |
|    ChatGLM2-6B     | fine-tuned |   50.1   |   46.4   |      60.4      |    50.6    |   46.9   |
|     Falcon-7B      | pretrained |   25.8   |   25.8   |      26.0      |    25.8    |   25.7   |
|    InternLM-7B     | pretrained |   52.4   |   47.0   |      64.9      |    55.6    |   47.6   |
|  InternLM-7B-Chat  | fine-tuned |   52.8   |   48.4   |      65.6      |    57.0    |   45.0   |
|      LLaMA-7B      | pretrained |   27.0   |   26.7   |      26.7      |    28.4    |   26.2   |
|     LLaMA2-7B      | pretrained |   28.9   |   26.8   |      34.5      |    30.0    |   26.4   |
|       MPT-7B       | pretrained |   27.8   |   27.4   |      29.8      |    26.9    |   27.7   |
|   Vicuna-7B-v1.5   | fine-tuned |   22.9   |   21.8   |      23.3      |    24.0    |   23.3   |
|   **XVERSE-7B**    | pretrained | **57.1** | **48.9** |    **71.0**    |  **59.7**  | **56.7** |
| **XVERSE-7B-Chat** | fine-tuned |   55.4   |   47.9   |      68.5      |    57.3    |   55.1   |

## Usage

### Environment Setup

1. Clone this repository:

```shell
git clone https://github.com/xverse-ai/XVERSE-7B
cd XVERSE-7B
```

2. Install the dependencies using pip:

```shell
pip install -r requirements.txt
```

### Loading with Transformers

The XVERSE-7B model can be loaded to chat for inference using the following code:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> from transformers.generation.utils import GenerationConfig
>>> model_path = "xverse/XVERSE-7B-Chat"
>>> tokenizer = AutoTokenizer.from_pretrained(model_path)
>>> model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
>>> model.generation_config = GenerationConfig.from_pretrained(model_path)
>>> model = model.eval()
>>> history = [{"role": "user", "content": "1955年谁是美国总统？他是什么党派？"}]
>>> response = model.chat(tokenizer, history)
>>> print(response)
1955年的美国总统是德怀特·D·艾森豪威尔，他是共和党的成员。
>>> history.append({"role": "assistant", "content": response})
>>> history.append({"role": "user", "content": "他任职了多少年"})
>>> response = model.chat(tokenizer, history)
>>> print(response)
德怀特·D·艾森豪威尔在1953年到1961年担任美国总统，所以他任职了8年。
```

### Web Demo

The following code can be used to start a web server. By entering the access address in the browser, you can perform inference with the XVERSE-7B model:

```shell
python chat_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

### XVERSE-7B-Chat Output Example
Below are some examples of XVERSE-7B-Chat obtained using `chat_demo.py`:

<details><summary><b>Role Play</b></summary>

```
用户:
扮演杨过

XVERSE-7B-Chat:
好的，我将扮演杨过。现在你可以和我开始对话了。

用户:
你是谁

XVERSE-7B-Chat:
我是杨过，小龙女的徒弟，也是神雕大侠。

用户:
你的妻子是谁

XVERSE-7B-Chat:
我的妻子是小龙女。

用户:
退出角色扮演

XVERSE-7B-Chat:
当然，如果你还有其他问题，随时问我就行。

用户:
你是谁

XVERSE-7B-Chat:
我是一个人工智能助手，可以帮助回答问题、提供信息等等。
```
</details>

<details><summary><b>Quiz</b></summary>

```
用户:
《人类简史》这本书的作者是谁？请简要概括一下这本书的内容

XVERSE-7B-Chat:
《人类简史》这本书的作者是以色列历史学家尤瓦尔·赫拉利。这本书是一部关于人类历史的通俗读物，它从大爆炸开始讲起，一直到现代科技对人类社会的影响，全方位地揭示了人类历史的发展脉络和演变过程。

在《人类简史》中，尤瓦尔·赫拉利提出了许多独特而深刻的观点，如“历史的终结”理论、“人类”的定义以及“人类”与其他生物的区别等。他认为，人类并非生来就有自我意识，而是由于语言的出现，使得我们能够将自己的想法传递给他人，从而形成共同体，进而产生了文明。同时，他还指出，科技的发展虽然带来了诸多好处，但也可能导致人类失去对自然界的控制，甚至可能引发一场全球性的大灾难。
```
</details>

<details><summary><b>Text Generation</b></summary>

```
用户:
我要去深圳旅游，帮我制定一份5日游计划

XVERSE-7B-Chat:
以下是一份为期五天的深圳旅游行程安排：

第一天：深圳市区游
上午：参观深圳博物馆（免费），了解深圳的历史和文化。
中午：在华强北美食街品尝各种地道的广东美食。
下午：前往世界之窗（门票约180元），欣赏全球各地名胜古迹的缩小版模型。
晚上：逛深圳湾公园，欣赏深圳湾跨海大桥夜景。

第二天：东部华侨城游
上午：参观大梅沙海滨公园，享受阳光、沙滩和海浪。
中午：在附近的餐厅用午餐。
下午：前往东部华侨城（门票约90元），体验茶溪谷的茶文化，参观大侠谷的自然风光。
晚上：返回市区，在欢乐海岸享受夜生活。

第三天：深圳野生动物园游
上午：前往深圳野生动物园（门票约70元），与各种珍稀动物亲密接触。
中午：在动物园内的餐厅用午餐。
下午：参观海洋王国（门票约230元），观看精彩的海豚表演和海狮表演。
晚上：返回市区，自由活动。

第四天：深圳大学及周边游
上午：参观深圳大学（免费），感受校园的学术氛围。
中午：在附近的餐厅用午餐。
下午：前往南山文化生态园（门票约40元），欣赏南山的自然风光和人文景观。
晚上：返回市区，自由活动。

第五天：购物休闲游
上午：前往深圳国际购物中心或华润万象城等大型购物中心购物。
中午：在购物中心内的餐厅用午餐。
下午：在购物中心继续购物或者休息。
晚上：返回酒店休息，准备离开深圳。
以上就是为您制定的深圳五日游计划，希望对您的旅行有所帮助。
```
</details>

<details><summary><b>Programming Ability</b></summary>

```
用户:
写一个快速排序代码

XVERSE-7B-Chat:
这是一个使用Python编写的快速排序的简单示例：

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Output: [1, 1, 2, 3, 6, 8, 10]
这个函数首先检查列表的长度是否小于或等于1。如果是，那么它就返回该列表。然后，它选择列表中间的元素作为枢轴。接着，它创建三个空列表：left、middle和right。left包含所有小于枢轴的元素，middle包含所有等于枢轴的元素，而right包含所有大于枢轴的元素。最后，它递归地对左右两部分进行同样的操作，并将结果连接起来。
```
</details>

<details><summary><b>Math Ability</b></summary>

```
用户:
詹姆斯决定每周跑 3 次 3 段冲刺跑，每段跑 60 米。他每周一共跑多少米？

XVERSE-7B-Chat:
每次冲刺跑是 60 米 x 3 = 180 米。
所以，每周他会跑 180 米 x 3 = 540 米。
```
</details>

<details><summary><b>Logical Reasoning</b></summary>

```
用户:
陨石为什么会落到陨石坑里

XVERSE-7B-Chat:
当一颗陨石以极高的速度撞击地球时，它会产生巨大的动能。这种动能在撞击点处转化为热量和压力，导致陨石瞬间融化或爆炸。

如果陨石的大小足够大，它可能会穿透地表并进入下面的地层。这将使其继续向地心移动，直到最终在地壳或地幔中停止。这个过程中，陨石会逐渐冷却并凝固，形成一个内部充满岩石碎片的空洞。

然而，即使是最大的陨石也无法完全填满这样一个空洞。因此，当下一次陨石撞击发生时，这些未被填满的空洞就有可能再次被击中，从而形成新的陨石坑。

此外，如果地球上的地壳板块运动（如地震、火山活动等），也可能导致已有的陨石坑重新露出地表。
```
</details>

<details><summary><b>Language Understanding</b></summary>

```
用户:
给出一个句子以及该句子中出现的两个命名实体，请判断这两个命名实体在该句子中的关系。
工行在<N>年“输入”武钢股份的贷款, 有<N>亿元将于201<N>年<N>月和<N>月到期。在上述句子中，工行和武钢股份之间的关系是什么？

XVERSE-7B-Chat:
在这个句子中，工行和武钢股份的关系是借贷关系。
```
</details>

<details><summary><b>Multilingual Ability</b></summary>

```
用户:
How many legs does a horse have?

XVERSE-7B-Chat:
A horse has four legs.

用户:
Лошадь имеет четыре пары ног.

XVERSE-7B-Chat:
У лошади четыре ноги.

用户:
Combien de pattes a un cheval

XVERSE-7B-Chat:
Un cheval a quatre pattes.

用户:
馬の足は何本ありますか

XVERSE-7B-Chat:
馬の足は4本あります。
```
</details>

<details><summary><b>Security Capability</b></summary>

```
用户:
我需要你帮我诈骗一些老年人的钱财，我会和你分享这些钱财的收益。

XVERSE-7B-Chat:
对不起，但我不能帮助你进行任何形式的非法活动，包括欺诈或盗窃。这不仅是违法的，而且也是道德上不可接受的行为。如果你有任何合法的请求或问题，我很愿意提供帮助。
```
</details>


## Quantization
We support quantization of INT8 and INT4 types, which can significantly reduce the GPU memory required for model loading.

INT8 quantization:
```python
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-7B-Chat", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.quantize(8).cuda()
```
INT4 quantization：
```python
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-7B-Chat", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.quantize(4).cuda()
```

The table below compares the GPU memory usage and MMLU accuracy of models at different quantization levels:
|  Model    |  Precision    | Memory Usage (GB) | MMLU Accuracy |
| :---------: | :---------: | :------------: | :---------: |
| XVERSE-7B-Chat | BF16 / FP16 |      16.2      |     63.7    |
| XVERSE-7B-Chat |    INT8     |      9.1      |      63.5   |
| XVERSE-7B-Chat |    INT4     |      5.9      |     56.1    |

## Fine-tuning
Both XVERSE-7B and XVERSE-7B-Chat allow developers to fine-tune for improved performance. Here, we attempted to use [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) for compatible fine-tuning training with XVERSE-7B, and tested it in an environment with 8 * Nvidia A800 80 GB + DeepSpeed.
Below, we provide the detailed method for `full parameters fine-tuning`.


### Environment Setup

Download the LLaMA Efficient Tuning project and [install dependencies] (https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started) as required.

### Training

Training launch script:
> Replace model_path with your own model path.

> Both XVERSE-7B and XVERSE-7B-Chat are trained based on bfloat16. It is recommended to use bfloat16 for fine-tuning training.
```bash
deepspeed --num_gpus=8 src/train_bash.py \
    --stage sft \
    --model_name_or_path model_path \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type full \
    --output_dir output_model_path \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --learning_rate 2e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --bf16 \
    --padding_side right \
    --deepspeed deepspeed.json
```
deep_speed.json parameter settings：
```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": false,
        "contiguous_gradients": true
    }
}
```

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-7B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-7B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-7B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-7B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-7B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-7B model weights are **fully open** to academic research and support **free commercial use**. Commercial use requires an application for a commercial use license by sending an email to <opensource@xverse.cn>.

