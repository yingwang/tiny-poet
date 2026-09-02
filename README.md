# tiny-poet

A minimal GPT implemented from scratch and trained on the Complete Tang Poems and the Complete Song Ci, capable of generating classical Chinese verse.

[中文](#tiny-poet-1)

The core model and training loop are self-contained and run to fewer than 500 lines. It was written with three objectives in mind:

- Every line of the transformer should be legible to the reader
- Training should complete on a free Colab T4
- The trained model should produce classical verse of reasonable quality

## Model Sizes

| Config | Params | Colab T4 Training Time |
|--------|--------|------------------------|
| tiny | 1.71M | 20-30 min |
| small | 6.57M | 1-2 hrs |
| base | 16.99M | 3-5 hrs |

Parameter counts are given for the actual vocabulary of 7,098 characters that `data.py` now produces (the v0.1 release was trained on 11,601; see Data). The token embedding accounts for a substantial share of a character-level model at this scale, so these figures run higher than the 6,000-vocabulary default in `GPTConfig` would suggest.

The default configuration is `small`. At this dataset size `small` and `base` perform comparably; `base` overfits somewhat more, but its output carries a stronger classical register.

## Quick Start

To generate directly from a released checkpoint, without training:

```bash
pip install -r requirements.txt
wget https://github.com/yingwang/tiny-poet/releases/download/v0.2/small_inference.pt
python sample.py --ckpt small_inference.pt --prompt "春" --num_samples 3
```

Release v0.2 is the `small` configuration at 6.57M parameters on the current 7,098-character simplified vocabulary, trained for 10,000 iterations (validation loss 4.15). The earlier v0.1 checkpoint (7.72M parameters, the mixed-script 11,601-character vocabulary, final loss 4.84) remains downloadable from the v0.1 release. Every checkpoint carries its own vocabulary, so old and new checkpoints both load; `sample.py` converts the prompt to the script the checkpoint was trained on.

## Training from Scratch

### Local (iMac / MacBook)

```bash
pip install -r requirements.txt

# 1. Download and preprocess the pinned corpus (Complete Tang Poems + Song Ci)
python data.py

# 2. Train the tiny configuration (roughly 2-4 hours on an iMac CPU)
python train.py --config tiny --device cpu --iters 5000

# 3. Generate from the best validation checkpoint
python sample.py --ckpt checkpoints/tiny_best.pt --prompt "春眠不觉晓" --max_tokens 50

# 4. Optional: strip the optimizer state to produce a distributable checkpoint
python export.py --ckpt checkpoints/tiny_best.pt
```

### Colab

```bash
!git clone https://github.com/yingwang/tiny-poet.git
%cd tiny-poet
!pip install -r requirements.txt

!python data.py
!python train.py --config small --iters 10000
!python sample.py --ckpt checkpoints/small_best.pt --prompt "春眠不觉晓" --max_tokens 50
```

`train.py` selects the available accelerator automatically (cuda / mps / cpu).

Training keeps separate checkpoints for separate jobs:

- `checkpoints/<config>_latest.pt` — latest training state, including RNG state; used by `--resume`
- `checkpoints/<config>_best.pt` — lowest validation loss; recommended for generation and export
- `checkpoints/<config>_final.pt` — state at the requested final iteration

`--iters` is the total target iteration count. For example, after a 10,000-iteration run, continue to 15,000 with:

```bash
python train.py --config small --iters 15000 --resume
```

Run the CPU test suite with:

```bash
python -m unittest discover -s tests -v
```

## Files

- `data.py` — Downloads the chinese-poetry corpus, normalizes its script, drops duplicate works, and builds the character vocabulary
- `model.py` — GPT architecture: embedding → N × transformer block → output
- `train.py` — Training loop: AdamW with a cosine schedule, checkpoint support
- `sample.py` — Inference by top-k sampling
- `export.py` — Strips the optimizer state to produce a release-sized checkpoint
- `tests/` — CPU tests for data splitting, generation, checkpointing, and resume

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

Source: [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry), pinned to commit [`b8594f8`](https://github.com/chinese-poetry/chinese-poetry/commit/b8594f81a89752241442f2ce267d6f66f96704ee) for reproducible preprocessing.

Preprocessing retries failed files and stops rather than silently creating a partial corpus. Use `python data.py --allow-partial` only when an incomplete dataset is intentional.

The upstream corpus is not in one script: the Tang poems are stored in traditional characters and the Song ci in simplified. Trained on that mixture, a character-level model learns that traditional means shi and simplified means ci, and mixes both scripts inside one poem (the v0.1 samples below show it). `data.py` therefore converts everything to simplified with OpenCC (`--script traditional` converts the other way; `--script mixed` keeps the upstream text). It also drops later copies of a work whose body already appeared, punctuation aside (984 of them, which would otherwise straddle the train/validation split), and files each Song ci under its tune name alone, without the author, so the model stops following every tune with an invented name.

- Complete Tang Poems: 57,435 pieces
- Complete Song Ci: 21,050 pieces
- After de-duplication: 77,501 works, 6.07M characters (95% train, 5% validation, split by whole works)

Character-level tokenizer. Characters seen fewer than three times in the training text (`--min-count`) map to one unknown token, `□`; that removes about 4,500 characters that appeared once or twice in six million and leaves a vocabulary of 7,098. The unknown token covers 0.14% of the tokens.

## Sample Output (v0.2 small)

Samples from the v0.2 release checkpoint with `--seed 7`, `--temperature 0.9`, `--top_k 40`. Everything after the prompt, including the titles and tune names, is generated.

Prompt `春`:
> 春愁。莫言千里去，莫遣一江流。
> 题郑谷府公主园林
> 地与山川旧隐居，闲门竹竹似经书。何当世事知心静，犹得清光独赏初。

Prompt `月`:
> 月高
> 风高一望是重阳，千里孤光万里乡。一点秋山过江岸，碧云红树在东央。
> 寄远十九
> 忆昔年来几度春，见君犹得独为人。

Prompt `江南`:
> 江南・忆江南
> 忆去日、春风无数。日转小池亭下草。柳色浅、春光如旧。东风只道青门远。回首两行愁绪在。花外雨、酒边烟。

Prompt `春眠不觉晓`:
> 春眠不觉晓，云影几千间。
> 送友人游江南
> 春雨一帆斜照沙，江山万里隔烟霞。归来好鸟无家住，为见春愁一点瓜。

One script throughout and no invented author after a tune name, both of which v0.1 got wrong (its corpus mixed traditional Tang poems with simplified Song ci, and filed each ci under tune and author). The corpus still contains the 兵要望江南 divination ci with their collation notes (`京本作…`), which the model occasionally reproduces; filtering those titles out is the obvious next data step. No systematic memorization audit has been performed, so generated text should be checked against the source corpus before publication.

## License

Project code is released under the MIT License. The downloaded chinese-poetry corpus is also distributed under its [upstream MIT License](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE).

---

# tiny-poet

一个从零实现的小型 GPT，以全唐诗与全宋词训练，能够生成古体诗词。

核心模型与训练循环自成一体，代码总量不足 500 行。写作之初设定了三项目标：

- transformer 的每一行实现都应当清晰可读
- 训练能够在免费的 Colab T4 上完成
- 训练完成后能够生成具有一定水准的诗词

## 模型规模

| 配置 | 参数量 | Colab T4 训练时间 |
|------|--------|-------------------|
| tiny | 1.71M | 20-30分钟 |
| small | 6.57M | 1-2小时 |
| base | 16.99M | 3-5小时 |

参数量按 `data.py` 现在产出的实际词表 7,098 计算（v0.1 发布版训练时词表为 11,601，见「数据」一节）。字符级模型在这一量级上 token embedding 占比较大，因此该数值高于按 `GPTConfig` 中默认的 6,000 词表所得的结果。

默认配置为 `small`。在当前数据规模下，`small` 与 `base` 表现相近，`base` 的过拟合略为明显，但生成结果的古典气息更浓。

## 快速开始（使用已训练的模型）

若希望跳过训练环节，直接从已发布的 checkpoint 生成：

```bash
pip install -r requirements.txt
wget https://github.com/yingwang/tiny-poet/releases/download/v0.2/small_inference.pt
python sample.py --ckpt small_inference.pt --prompt "春" --num_samples 3
```

v0.2 采用 `small` 配置，参数量 6.57M，词表为当前的简体版本（7,098 字），训练 10,000 步，验证 loss 4.15。此前的 v0.1 checkpoint（7.72M 参数、简繁混杂的 11,601 字词表、最终 loss 4.84）仍可从 v0.1 发布页下载。每个 checkpoint 都自带词表，新旧版本都能加载；`sample.py` 会把提示词转换成 checkpoint 训练时所用的字体。

## 从零训练

### 本地（iMac / MacBook）

```bash
pip install -r requirements.txt

# 1. 下载并预处理已锁定版本的语料（全唐诗 + 全宋词）
python data.py

# 2. 训练 tiny 配置（iMac CPU 约需 2 至 4 小时）
python train.py --config tiny --device cpu --iters 5000

# 3. 使用验证集最优 checkpoint 生成
python sample.py --ckpt checkpoints/tiny_best.pt --prompt "春眠不觉晓" --max_tokens 50

# 4. 可选：剥离 optimizer state，得到便于分发的 checkpoint
python export.py --ckpt checkpoints/tiny_best.pt
```

### Colab

```bash
!git clone https://github.com/yingwang/tiny-poet.git
%cd tiny-poet
!pip install -r requirements.txt

!python data.py
!python train.py --config small --iters 10000
!python sample.py --ckpt checkpoints/small_best.pt --prompt "春眠不觉晓" --max_tokens 50
```

`train.py` 会自动选择可用的计算设备（cuda / mps / cpu）。

训练过程分别保存三类 checkpoint：

- `checkpoints/<config>_latest.pt`：最新训练状态，包含随机数状态；供 `--resume` 使用
- `checkpoints/<config>_best.pt`：验证 loss 最低的状态；推荐用于生成和导出
- `checkpoints/<config>_final.pt`：达到目标 iteration 时的最终状态

`--iters` 表示目标总 iteration 数。例如完成 10,000 次后继续训练到 15,000 次：

```bash
python train.py --config small --iters 15000 --resume
```

运行 CPU 测试：

```bash
python -m unittest discover -s tests -v
```

## 文件说明

- `data.py`：下载 chinese-poetry 语料，统一字体，去除重复作品，构建字符级词表
- `model.py`：GPT 架构，embedding → N × transformer block → output
- `train.py`：训练循环，AdamW 配合 cosine schedule，支持 checkpoint
- `sample.py`：推理，采用 top-k 采样
- `export.py`：剥离 optimizer state，产出发布用的精简 checkpoint
- `tests/`：覆盖数据切分、生成、checkpoint 与续训的 CPU 测试

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

来源：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，预处理固定到 commit [`b8594f8`](https://github.com/chinese-poetry/chinese-poetry/commit/b8594f81a89752241442f2ce267d6f66f96704ee)，确保结果可复现。

预处理会重试失败文件，并默认拒绝生成残缺语料。只有明确需要不完整数据集时才使用 `python data.py --allow-partial`。

上游语料并非同一种字体：全唐诗以繁体存储，全宋词以简体存储。直接在这样的混合语料上训练，字符级模型会学到「繁体即诗、简体即词」的假规律，并在同一首作品里混用两种字体（下方 v0.1 的样本即是如此）。因此 `data.py` 用 OpenCC 把全部文本统一为简体（`--script traditional` 反向转换，`--script mixed` 保留上游原文）。它还会按正文去重，标点不计（共 984 首，否则同一首诗会同时落在训练集和验证集两侧），并把每首宋词只按词牌归档、不带作者，模型也就不再在每个词牌后面编造一个人名。

- 全唐诗：57,435 首
- 全宋词：21,050 首
- 去重后：77,501 首，607 万字（按整首切分，95% 训练、5% 验证）

字符级 tokenizer。训练文本中出现不足三次的字（`--min-count`）映射为一个未知符号 `□`，由此去掉约 4,500 个在六百万字里只出现一两次的字，词表规模为 7,098。未知符号占全部 token 的 0.14%。

## 样本输出（v0.2 small）

以下样本来自 v0.2 发布的 checkpoint，参数为 `--seed 7`、`--temperature 0.9`、`--top_k 40`。提示词之后的一切，包括标题与词牌，都由模型生成。

输入 `春`：
> 春愁。莫言千里去，莫遣一江流。
> 题郑谷府公主园林
> 地与山川旧隐居，闲门竹竹似经书。何当世事知心静，犹得清光独赏初。

输入 `月`：
> 月高
> 风高一望是重阳，千里孤光万里乡。一点秋山过江岸，碧云红树在东央。
> 寄远十九
> 忆昔年来几度春，见君犹得独为人。

输入 `江南`：
> 江南・忆江南
> 忆去日、春风无数。日转小池亭下草。柳色浅、春光如旧。东风只道青门远。回首两行愁绪在。花外雨、酒边烟。

输入 `春眠不觉晓`：
> 春眠不觉晓，云影几千间。
> 送友人游江南
> 春雨一帆斜照沙，江山万里隔烟霞。归来好鸟无家住，为见春愁一点瓜。

全文只有一种字体，词牌之后不再出现编造的作者名；这两点 v0.1 都没做到，因为它的语料把繁体的全唐诗与简体的全宋词混在一起，并把每首词按「词牌·作者」归档。语料中仍有《兵要望江南》一类带校勘注释（`京本作…`）的军占词，模型偶尔会复现，把这类标题从语料中剔除是下一步显而易见的数据工作。目前尚未进行系统的记忆性审计，公开使用生成文本前应与源语料核对。

## 许可证

项目代码采用 MIT License。下载的 chinese-poetry 语料亦采用其[上游 MIT License](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE)。
