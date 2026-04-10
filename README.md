# tiny-poet

一个从零实现的小型 GPT，用全唐诗 + 全宋词训练，能生成古体诗词。

单文件实现（除了数据和 tokenizer），代码不到 500 行。目标是：
- 看得懂 transformer 每一行
- 能在 Colab 免费 T4 上训练
- 训完能真的生成像样的诗词

## 模型规模

| 配置 | 参数量 | Colab T4 训练时间 |
|------|--------|-------------------|
| tiny | ~1.6M | 20-30分钟 |
| small | ~6.3M | 1-2小时 |
| base | ~16.6M | 3-5小时 |

默认用 `small`。对唐诗宋词这个数据量，`small` 和 `base` 差别不大，`base` 稍微过拟合一点但生成更有古味。

## 快速开始（用训好的模型）

想跳过训练直接玩生成？下载发布好的 checkpoint：

```bash
pip install torch numpy
wget https://github.com/yingwang/tiny-poet/releases/download/v0.1/small_inference.pt
python sample.py --ckpt small_inference.pt --prompt "春" --num_samples 3
```

v0.1 是 small 配置 7.72M 参数，在 iMac 2019 上训了 90 分钟，final loss 4.84。

## 从零训练

### 本地（iMac / MacBook）

```bash
pip install torch numpy tqdm

# 1. 下载数据（全唐诗 + 全宋词）
python data.py

# 2. 训练 tiny 配置（iMac CPU 2-4 小时能出效果）
python train.py --config tiny --device cpu --iters 5000

# 3. 生成
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

## 文件说明

- `data.py` — 下载并清洗 chinese-poetry 数据集（字符级）
- `model.py` — GPT 架构：embedding → N × transformer block → output
- `train.py` — 训练循环：AdamW + cosine schedule，支持 checkpoint
- `sample.py` — 推理：top-k 采样生成

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

标准 GPT-style decoder-only transformer，没有花里胡哨。

## 数据

来源：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- 全唐诗：约 55k 首
- 全宋词：约 21k 首
- 总 token 数：约 500 万字符

字符级 tokenizer，实际 vocab size 11,601（简体 + 繁体 + 标点 + 少量异体字）。

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

（作者名是模型自己造的，词句大部分也是新生成而非训练数据原文）
