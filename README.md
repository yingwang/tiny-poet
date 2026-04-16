# tiny-poet

一个从零实现的小型 GPT，用全唐诗 + 全宋词训练，能生成古体诗词。

A tiny GPT built from scratch, trained on the Complete Tang Poems + Complete Song Ci, capable of generating classical Chinese poetry.

单文件实现（除了数据和 tokenizer），代码不到 500 行。目标是：
- 看得懂 transformer 每一行
- 能在 Colab 免费 T4 上训练
- 训完能真的生成像样的诗词

Single-file implementation (aside from data and tokenizer), under 500 lines of code. Goals:
- Understand every line of the transformer
- Train on a free Colab T4
- Actually generate decent classical poetry

## 模型规模 / Model Sizes

| 配置 Config | 参数量 Params | Colab T4 训练时间 Training Time |
|-------------|---------------|--------------------------------|
| tiny | ~1.6M | 20-30 min |
| small | ~6.3M | 1-2 hrs |
| base | ~16.6M | 3-5 hrs |

默认用 `small`。对唐诗宋词这个数据量，`small` 和 `base` 差别不大，`base` 稍微过拟合一点但生成更有古味。

Default is `small`. For this dataset size, `small` and `base` perform similarly — `base` overfits slightly more but produces text with a stronger classical flavor.

## 快速开始 / Quick Start

想跳过训练直接玩生成？下载发布好的 checkpoint：

Want to skip training and jump straight to generation? Download a pre-trained checkpoint:

```bash
pip install torch numpy
wget https://github.com/yingwang/tiny-poet/releases/download/v0.1/small_inference.pt
python sample.py --ckpt small_inference.pt --prompt "春" --num_samples 3
```

v0.1 是 small 配置 7.72M 参数，在 iMac 2019 上训了 90 分钟，final loss 4.84。

v0.1 uses the `small` config with 7.72M parameters, trained on an iMac 2019 for 90 minutes, final loss 4.84.

## 从零训练 / Training from Scratch

### 本地 / Local (iMac / MacBook)

```bash
pip install torch numpy tqdm

# 1. 下载数据（全唐诗 + 全宋词）/ Download data (Complete Tang Poems + Song Ci)
python data.py

# 2. 训练 tiny 配置 / Train with tiny config (iMac CPU 2-4 hrs)
python train.py --config tiny --device cpu --iters 5000

# 3. 生成 / Generate
python sample.py --prompt "春眠不觉晓" --max_tokens 50
```

### Colab

```bash
!git clone https://github.com/yingwang/tiny-poet.git
%cd tiny-poet
!pip install torch numpy tqdm

!python data.py
!python train.py --config small --iters 10000
!python sample.py --prompt "春眠不觉晓" --max_tokens 50
```

`train.py` 会自动检测 GPU（cuda / mps / cpu）。

`train.py` auto-detects GPU (cuda / mps / cpu).

## 文件说明 / Files

- `data.py` — 下载并清洗 chinese-poetry 数据集（字符级）/ Downloads and cleans the chinese-poetry dataset (character-level)
- `model.py` — GPT 架构：embedding → N × transformer block → output / GPT architecture
- `train.py` — 训练循环：AdamW + cosine schedule，支持 checkpoint / Training loop with checkpoint support
- `sample.py` — 推理：top-k 采样生成 / Inference: top-k sampling

## 架构 / Architecture

```
Input (char ids)
  ↓ Token Embedding + Positional Embedding
  ↓
  [Transformer Block] × N
    ├── LayerNorm
    ├── Multi-Head Self-Attention (causal)
    ├── LayerNorm
    └── MLP (4x hidden)
  ↓
LayerNorm
  ↓ Linear → vocab_size
Softmax → next char probabilities
```

标准 GPT-style decoder-only transformer，没有花里胡哨。

Standard GPT-style decoder-only transformer. No bells and whistles.

## 数据 / Data

来源 / Source: [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- 全唐诗 Complete Tang Poems: ~55k
- 全宋词 Complete Song Ci: ~21k
- 总 token 数 Total tokens: ~5M characters

字符级 tokenizer，实际 vocab size 11,601（简体 + 繁体 + 标点 + 少量异体字）。

Character-level tokenizer, vocab size 11,601 (simplified + traditional Chinese + punctuation + variant characters).

## 样本输出 / Sample Output (v0.1 small)

输入 / Prompt `春`:
> 春意，柳阴如雨。春似故人来醉。
> 送友客
> 別離辭別，春風欲多。白髮相逢客，寒枝半似春。

输入 / Prompt `月`:
> 月·沈丘崈
> 一点春容不见。无人有酒。不似花梢柳。花如玉。梅花风，也似西西子。

输入 / Prompt `江南`:
> 江南·念奴娇·王安安岳
> 春云已暮，不怕风流水。云树碧流沙外，江外一声寒水。

（作者名是模型自己造的，词句大部分也是新生成而非训练数据原文）

(Author names are hallucinated by the model; most phrases are novel generations, not memorized training data.)
