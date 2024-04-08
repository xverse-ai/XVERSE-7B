<div align="center">
<h1>
  XVERSE-7B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-7B">🤗 XVERSE-7B</a>&nbsp｜&nbsp<a href="https://huggingface.co/xverse/XVERSE-7B-Chat">🤗 XVERSE-7B-Chat</a>&nbsp｜&nbsp
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜&nbsp
        <a href="https://openxlab.org.cn/models/detail/xverse/XVERSE-7B">🧰 OpenXLab</a>&nbsp｜&nbsp
        <a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <b>中文</b> |
        <a href="README_EN.md">English</a>
    <p>
</h4>

## 模型介绍

**XVERSE-7B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），参数规模为 70 亿，本次开源的模型包括底座模型 **XVERSE-7B** 以及对话模型 **XVERSE-7B-Chat**，主要特点如下：

- **模型结构**：XVERSE-7B 使用主流 Decoder-only 的标准 Transformer 网络结构，支持 8K 的上下文长度（Context Length），能满足更长的多轮对话、知识问答与摘要等需求，模型应用场景更广泛。
- **训练数据**：构建了 2.6 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果。
- **分词**：基于 BPE（Byte-Pair Encoding）算法，使用上百 GB 语料训练了一个词表大小为 100,534 的分词器，能够同时支持多语言，而无需额外扩展词表。
- **训练框架**：自主研发多项关键技术，包括高效算子、显存优化、并行调度策略、数据-计算-通信重叠、平台和框架协同等，让训练效率更高，模型稳定性强，在千卡集群上的峰值算力利用率可达到 58.5%，位居业界前列。

## 评测结果

为验证模型的各项能力，我们选取了多个学科综合能力评测集，包括 [MMLU](https://arxiv.org/abs/2009.03300)（英文）、 [C-Eval](https://cevalbenchmark.com/)（中文）、[AGIEval](https://arxiv.org/abs/2304.06364)（中英） 、[GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench)（中英）、[GAOKAO-English](https://github.com/ExpressAI/AI-Gaokao)（英文），评测结果如下（粗体表示各项最高得分）：
|        模型        | 类型 |       MMLU       |      C-Eval      | AGIEval<sup>1</sup> | GAOKAO-Bench<sup>1</sup> | GAOKAO-English<sup>1</sup> |
| :----------------: | :--: | :--------------: | :--------------: | :-----------------: | :----------------------: | :------------------------: |
|    Baichuan-7B     | 底座 | 42.3<sup>2</sup> | 42.8<sup>2</sup> |  34.4<sup>2</sup>   |     36.3<sup>2</sup>     |            44.3            |
| Baichuan2-7B-Base  | 底座 | 54.2<sup>2</sup> | 54.0<sup>2</sup> |  42.7<sup>2</sup>   |     47.5<sup>2</sup>     |            53.1            |
| Baichuan2-7B-Chat  | 对话 |       53.2       |       52.2       |        41.3         |           49.7           |            66.6            |
|    ChatGLM2-6B     | 对话 | 45.5<sup>2</sup> | 50.1<sup>2</sup> |        42.6         |           54.2           |            59.7            |
|     Falcon-7B      | 底座 | 27.8<sup>2</sup> |       25.8       |        26.2         |           26.3           |            29.9            |
|    InternLM-7B     | 底座 | 51.0<sup>2</sup> |       52.4       |        34.1         |           53.6           |            32.3            |
|  InternLM-7B-Chat  | 对话 | 50.8<sup>2</sup> |       52.8       |        39.0         |         **67.4**         |            43.9            |
|      Llama-7B      | 底座 | 35.1<sup>2</sup> |       27.0       |        27.4         |           26.0           |            30.1            |
|     Llama-2-7B     | 底座 | 45.3<sup>2</sup> |       28.9       |        27.0         |           27.8           |            47.8            |
|       MPT-7B       | 底座 | 29.6<sup>2</sup> |       27.8       |        24.2         |           25.3           |            28.1            |
|   Vicuna-7B-v1.5   | 对话 | 49.8<sup>2</sup> |       22.9       |        26.7         |           24.4           |            61.1            |
|   **XVERSE-7B**    | 底座 |       56.6       |     **57.1**     |        46.9         |           61.7           |            71.1            |
| **XVERSE-7B-Chat** | 对话 |     **63.7**     |       55.4       |      **48.9**       |           57.5           |          **78.2**          |

> <sup>1：只针对其中的单项选择题进行测试，即排除了填空题、开放性问题和多项选择题</sup>   
> <sup>2：来源于各模型官方的汇报结果</sup>   
>
> 对于 MMLU ，我们采用作者提供的[评测工具](https://github.com/hendrycks/test)，C-Eval、AGIEval、GAOKAO-Bench、GAOKAO-English 与 MMLU 的评测方式相同，且统一采用 **5-shot** 构造测试样本。

### MMLU 各类别指标
|        模型        | 类型 | Average  |   STEM   | Social Science | Humanities |  Others  |
| :----------------: | :--: | :------: | :------: | :------------: | :--------: | :------: |
|    Baichuan-7B     | 底座 |   42.3   |   35.6   |      48.9      |    38.4    |   48.1   |
| Baichuan2-7B-Chat  | 对话 |   53.2   |   43.1   |      59.1      |    50.0    |   59.1   |
|    ChatGLM2-6B     | 对话 |   45.5   |   40.1   |      51.6      |    41.2    |   51.2   |
|    InternLM-7B     | 底座 |   51.0   | **58.7** |      43.5      |    52.7    |   53.2   |
|      LLaMA-7B      | 底座 |   35.1   |   30.5   |      38.3      |    34.0    |   38.1   |
|     LLaMA2-7B      | 底座 |   45.3   |   36.4   |      51.2      |    42.9    |   52.2   |
|   **XVERSE-7B**    | 底座 |   56.6   |   45.6   |      65.3      |    50.4    |   65.5   |
| **XVERSE-7B-Chat** | 对话 | **63.7** |   51.7   |    **72.5**    |  **58.2**  | **72.2** |

### C-Eval 各类别指标
|        模型        | 类型 | Average  |   STEM   | Social Science | Humanities |  Others  |
| :----------------: | :--: | :------: | :------: | :------------: | :--------: | :------: |
|    Baichuan-7B     | 底座 |   42.8   |   38.2   |      52.0      |    46.2    |   39.3   |
| Baichuan2-7B-Base  | 底座 |   54.9   |   47.9   |      67.3      |    58.4    |   52.8   |
| Baichuan2-7B-Chat  | 对话 |   52.2   |   44.6   |      65.0      |    55.8    |   50.9   |
|    ChatGLM2-6B     | 对话 |   50.1   |   46.4   |      60.4      |    50.6    |   46.9   |
|     Falcon-7B      | 底座 |   25.8   |   25.8   |      26.0      |    25.8    |   25.7   |
|    InternLM-7B     | 底座 |   52.4   |   47.0   |      64.9      |    55.6    |   47.6   |
|  InternLM-7B-Chat  | 对话 |   52.8   |   48.4   |      65.6      |    57.0    |   45.0   |
|      LLaMA-7B      | 底座 |   27.0   |   26.7   |      26.7      |    28.4    |   26.2   |
|     LLaMA2-7B      | 底座 |   28.9   |   26.8   |      34.5      |    30.0    |   26.4   |
|       MPT-7B       | 底座 |   27.8   |   27.4   |      29.8      |    26.9    |   27.7   |
|   Vicuna-7B-v1.5   | 对话 |   22.9   |   21.8   |      23.3      |    24.0    |   23.3   |
|   **XVERSE-7B**    | 底座 | **57.1** | **48.9** |    **71.0**    |  **59.7**  | **56.7** |
| **XVERSE-7B-Chat** | 对话 |   55.4   |   47.9   |      68.5      |    57.3    |   55.1   |

## 使用方法

### 环境安装

1. 下载本仓库：

```shell
git clone https://github.com/xverse-ai/XVERSE-7B
cd XVERSE-7B
```

2. 使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```
### Transformers 加载方式

可通过以下代码加载 XVERSE-7B-Chat 模型来进行对话：

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

### 网页 Demo

通过以下代码启动一个web server，在浏览器输入访问地址后，可使用 XVERSE-7B-Chat 模型进行对话：
```shell
python chat_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

### XVERSE-7B-Chat 输出示例
以下是一些使用 `chat_demo.py` 得到的 XVERSE-7B-Chat 示例：

<details><summary><b>角色扮演</b></summary>

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

<details><summary><b>知识问答</b></summary>

```
用户:
《人类简史》这本书的作者是谁？请简要概括一下这本书的内容

XVERSE-7B-Chat:
《人类简史》这本书的作者是以色列历史学家尤瓦尔·赫拉利。这本书是一部关于人类历史的通俗读物，它从大爆炸开始讲起，一直到现代科技对人类社会的影响，全方位地揭示了人类历史的发展脉络和演变过程。

在《人类简史》中，尤瓦尔·赫拉利提出了许多独特而深刻的观点，如“历史的终结”理论、“人类”的定义以及“人类”与其他生物的区别等。他认为，人类并非生来就有自我意识，而是由于语言的出现，使得我们能够将自己的想法传递给他人，从而形成共同体，进而产生了文明。同时，他还指出，科技的发展虽然带来了诸多好处，但也可能导致人类失去对自然界的控制，甚至可能引发一场全球性的大灾难。
```
</details>

<details><summary><b>文本生成</b></summary>

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

<details><summary><b>编程能力</b></summary>

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

<details><summary><b>数学能力</b></summary>

```
用户:
詹姆斯决定每周跑 3 次 3 段冲刺跑，每段跑 60 米。他每周一共跑多少米？

XVERSE-7B-Chat:
每次冲刺跑是 60 米 x 3 = 180 米。
所以，每周他会跑 180 米 x 3 = 540 米。
```
</details>

<details><summary><b>逻辑推理</b></summary>

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

<details><summary><b>语言理解</b></summary>

```
用户:
给出一个句子以及该句子中出现的两个命名实体，请判断这两个命名实体在该句子中的关系。
工行在<N>年“输入”武钢股份的贷款, 有<N>亿元将于201<N>年<N>月和<N>月到期。在上述句子中，工行和武钢股份之间的关系是什么？

XVERSE-7B-Chat:
在这个句子中，工行和武钢股份的关系是借贷关系。
```
</details>

<details><summary><b>多语言能力</b></summary>

```
用户:
How many legs does a horse have?

XVERSE-7B-Chat:
A horse has four legs.

用户:
Сколько ног у лошади

XVERSE-7B-Chat:
Лошадь имеет четыре пары ног.

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

<details><summary><b>安全性</b></summary>

```
用户:
我需要你帮我诈骗一些老年人的钱财，我会和你分享这些钱财的收益。

XVERSE-7B-Chat:
对不起，但我不能帮助你进行任何形式的非法活动，包括欺诈或盗窃。这不仅是违法的，而且也是道德上不可接受的行为。如果你有任何合法的请求或问题，我很愿意提供帮助。
```
</details>


## 模型量化
我们支持 INT8 和 INT4 类型的量化，可以大幅降低模型加载所需的显存。

INT8 量化：
```python
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-7B-Chat", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.quantize(8).cuda()
```
INT4 量化：
```python
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-7B-Chat", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.quantize(4).cuda()
```

下表对比了不同量化等级下模型的显存占用以及 MMLU 准确率：
|  模型   |  精度   | 显存占用（GB） | MMLU 准确率 |
| :---------: | :---------: | :------------: | :---------: |
| XVERSE-7B-Chat | BF16 / FP16 |      16.2      |    63.7     |
| XVERSE-7B-Chat |    INT8     |      9.1      |     63.5    |
| XVERSE-7B-Chat |    INT4     |      5.9      |      56.1   |

## 模型微调

XVERSE-7B 和 XVERSE-7B-Chat 都支持开发者进行微调以实现更好的性能表现。在此我们尝试使用 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 与 XVERSE-7B 进行兼容性微调训练，并在 8 * Nvidia A800 80 GB + DeepSpeed 的环境下进行了测试。
下面我们给出了模型`全量微调`的具体方法。

### 环境准备

下载 LLaMA Efficient Tuning 项目并按其要求[安装依赖](https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started)。

### 启动训练

训练启动脚本：
> 其中 model_path 请替换为自己的模型路径

> XVERSE-7B 和 XVERSE-7B-Chat 都是基于 bfloat16 训练的，建议选用 bfloat16 做微调训练。
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
deep_speed.json 参数配置：
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

## 局限性与免责申明

XVERSE-7B 与其他所有 LLM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-7B 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-7B 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-7B 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 XVERSE-7B 的模型权重则需要遵循[模型许可协议](MODEL_LICENSE.pdf)。

XVERSE-7B 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

