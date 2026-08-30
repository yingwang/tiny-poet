# tiny-poet

A minimal GPT implemented from scratch and trained on the Complete Tang Poems and the Complete Song Ci, capable of generating classical Chinese verse.

[中文](#tiny-poet-1)

The implementation is self-contained apart from the data pipeline and the tokenizer, and runs to fewer than 500 lines. It was written with three objectives in mind:

- Every line of the transformer should be legible to the reader
- Training should complete on a free Colab T4
- The trained model should produce classical verse of reasonable quality

## Model Sizes

| Config | Params | Colab T4 Training Time |
|--------|--------|------------------------|
| tiny | 2.29M | 20-30 min |
| small | 7.72M | 1-2 hrs |
| base | 18.72M | 3-5 hrs |

Parameter counts are given for the actual vocabulary of 11,601 characters. The token embedding accounts for a substantial share of a character-level model at this scale, so these figures run appreciably higher than the 6,000-vocabulary default in `GPTConfig` would suggest.

The default configuration is `small`. At this dataset size `small` and `base` perform comparably; `base` overfits somewhat more, but its output carries a stronger classical register.

## Quick Start

To generate directly from a released checkpoint, without training:

```bash
pip install torch numpy
wget https://github.com/yingwang/tiny-poet/releases/download/v0.1/small_inference.pt
python sample.py --ckpt small_inference.pt --prompt "春" --num_samples 3
```

Release v0.1 uses the `small` configuration at 7.72M parameters. It was trained on a 2019 iMac for 90 minutes, reaching a final loss of 4.84.

## Training from Scratch

### Local (iMac / MacBook)

```bash
pip install torch numpy

# 1. Download and preprocess the corpus (Complete Tang Poems + Song Ci)
python data.py

# 2. Train the tiny configuration (roughly 2-4 hours on an iMac CPU)
python train.py --config tiny --device cpu --iters 5000

# 3. Generate
python sample.py --prompt "春眠不觉晓" --max_tokens 50

# 4. Optional: strip the optimizer state to produce a distributable checkpoint
python export.py --ckpt checkpoints/tiny.pt
```

### Colab

```bash
!git clone https://github.com/yingwang/tiny-poet.git
%cd tiny-poet
!pip install torch numpy

!python data.py
!python train.py --config small --iters 10000
!python sample.py --prompt "春眠不觉晓" --max_tokens 50
```

`train.py` selects the available accelerator automatically (cuda / mps / cpu).

## Files

- `data.py` — Downloads and cleans the chinese-poetry corpus at the character level
- `model.py` — GPT architecture: embedding → N × transformer block → output
- `train.py` — Training loop: AdamW with a cosine schedule, checkpoint support
- `sample.py` — Inference by top-k sampling
- `export.py` — Strips the optimizer state to produce a release-sized checkpoint

## Architecture

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

A standard decoder-only transformer in the GPT style, without architectural embellishment.

## Data

Source: [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

- Complete Tang Poems: approximately 55,000 pieces
- Complete Song Ci: approximately 21,000 pieces
- Total characters: approximately 6.2M

Character-level tokenizer with a vocabulary of 11,601, covering simplified and traditional forms, punctuation, and a small number of variant characters.

## Sample Output (v0.1 small)

Prompt `春`:
> 春意，柳阴如雨。春似故人来醉。
> 送友客
> 別離辭別，春風欲多。白髮相逢客，寒枝半似春。

Prompt `月`:
> 月·沈丘崈
> 一点春容不见。无人有酒。不似花梢柳。花如玉。梅花风，也似西西子。

Prompt `江南`:
> 江南·念奴娇·王安安岳
> 春云已暮，不怕风流水。云树碧流沙外，江外一声寒水。

Author names are fabricated by the model. Most phrases are newly generated rather than reproduced from the training data.

---

# tiny-poet

一个从零实现的小型 GPT，以全唐诗与全宋词训练，能够生成古体诗词。

除数据处理与 tokenizer 之外为单文件实现，代码总量不足 500 行。写作之初设定了三项目标：

- transformer 的每一行实现都应当清晰可读
- 训练能够在免费的 Colab T4 上完成
- 训练完成后能够生成具有一定水准的诗词

## 模型规模

| 配置 | 参数量 | Colab T4 训练时间 |
|------|--------|-------------------|
| tiny | 2.29M | 20-30分钟 |
| small | 7.72M | 1-2小时 |
| base | 18.72M | 3-5小时 |

参数量按实际词表 11,601 计算。字符级模型在这一量级上 token embedding 占比较大，因此该数值明显高于按 `GPTConfig` 中默认的 6,000 词表所得的结果。

默认配置为 `small`。在当前数据规模下，`small` 与 `base` 表现相近，`base` 的过拟合略为明显，但生成结果的古典气息更浓。

## 快速开始（使用已训练的模型）

若希望跳过训练环节，直接从已发布的 checkpoint 生成：

```bash
pip install torch numpy
wget https://github.com/yingwang/tiny-poet/releases/download/v0.1/small_inference.pt
python sample.py --ckpt small_inference.pt --prompt "春" --num_samples 3
```

v0.1 采用 `small` 配置，参数量 7.72M，在 2019 款 iMac 上训练 90 分钟，最终 loss 为 4.84。

## 从零训练

### 本地（iMac / MacBook）

```bash
pip install torch numpy

# 1. 下载并预处理语料（全唐诗 + 全宋词）
python data.py

# 2. 训练 tiny 配置（iMac CPU 约需 2 至 4 小时）
python train.py --config tiny --device cpu --iters 5000

# 3. 生成
python sample.py --prompt "春眠不觉晓" --max_tokens 50

# 4. 可选：剥离 optimizer state，得到便于分发的 checkpoint
python export.py --ckpt checkpoints/tiny.pt
```

### Colab

```bash
!git clone https://github.com/yingwang/tiny-poet.git
%cd tiny-poet
!pip install torch numpy

!python data.py
!python train.py --config small --iters 10000
!python sample.py --prompt "春眠不觉晓" --max_tokens 50
```

`train.py` 会自动选择可用的计算设备（cuda / mps / cpu）。

## 文件说明

- `data.py`：下载并清洗 chinese-poetry 语料（字符级）
- `model.py`：GPT 架构，embedding → N × transformer block → output
- `train.py`：训练循环，AdamW 配合 cosine schedule，支持 checkpoint
- `sample.py`：推理，采用 top-k 采样
- `export.py`：剥离 optimizer state，产出发布用的精简 checkpoint

## 架构

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

标准的 GPT 式 decoder-only transformer，未作额外修饰。

## 数据

来源：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

- 全唐诗：约 55,000 首
- 全宋词：约 21,000 首
- 总字符数：约 620 万

字符级 tokenizer，词表规模 11,601，涵盖简体、繁体、标点以及少量异体字。

## 样本输出（v0.1 small）

输入 `春`：
> 春意，柳阴如雨。春似故人来醉。
> 送友客
> 別離辭別，春風欲多。白髮相逢客，寒枝半似春。

输入 `月`：
> 月·沈丘崈
> 一点春容不见。无人有酒。不似花梢柳。花如玉。梅花风，也似西西子。

输入 `江南`：
> 江南·念奴娇·王安安岳
> 春云已暮，不怕风流水。云树碧流沙外，江外一声寒水。

作者名系模型虚构，词句大多为新生成的内容，而非训练数据的原文。
