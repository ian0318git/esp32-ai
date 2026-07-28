# ESP32-AI — M5Stack CoreS3 雙語故事機

[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v6.0-blue)](https://github.com/espressif/esp-idf)
[![Target](https://img.shields.io/badge/target-ESP32--S3-orange)](https://www.espressif.com/en/products/socs/esp32-s3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🇹🇼 繁體中文** · [English](#english)

---

## 🇹🇼 繁體中文

### 概述

28.9M 參數語言模型在 **M5Stack CoreS3** 上完全離線運行。採用 Google Gemma 的 **Per-Layer Embeddings (PLE)** 技術，將 25M 參數的大表放進 Flash。支援 **中文** 與 **英文** 故事生成。

| | |
|---|---|
| 參數量 | 28.9M（25M 在 Flash 查找表） |
| 晶片 | ESP32-S3，512KB SRAM + 8MB PSRAM + 16MB Flash |
| 速度 | ~9.5 tok/s |
| 模型大小 | 14.9MB（4-bit 量化） |
| 訓練資料 | TinyStories（英文）/ TinyStoriesChinese（中文） |

### 功能特色

- ✅ **雙語支援** — 單一 Firmware 內建中英文詞彙表，開機按 `l` 切換
- ✅ **LCD 顯示中文** — M5GFX efontCN_16 字型，CJK 字符完整支援
- ✅ **隨機取樣** — 按 `t` 調整溫度（0.1~1.5），每次故事不同
- ✅ 10 個預設故事開頭（中英文各 10 個）
- ✅ 自訂 token ID 輸入（進階）
- ✅ CoreS3 ILI9342C 彩色 LCD（320x240）
- ✅ 全程 CPU 訓練，無 GPU 亦可完成

### 硬體需求

- **M5Stack CoreS3**
- USB-C 連接線

### 快速開始（雙語 Firmware）

```bash
# 清除 Flash（首次或變更分區後必做）
esptool --chip esp32s3 --port /dev/ttyACM0 erase-flash

# 燒錄雙語 Firmware（只需做一次）
arduino-cli upload -p /dev/ttyACM0 \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-dual-build \
  firmware/esp32_llm_dual

# 燒錄中文模型（0x1A0000）
source .venv/bin/activate
esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write-flash 0x1A0000 firmware/model_zh/model_zh.bin

# 監控
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

### 切換中英文

燒錄完成後，在序列埠選單按 **`l`** 即可切換：

```
===== PLE TinyLM =====
Select a prompt (0-9) or:
  l: Switch language
  temp=0.8 > l
语言: 中文

===== PLE TinyLM (中文) =====
选择故事开头 (0-9):
  l: 切换语言
  随机度=0.8 > 0
>>> 从前有一只小兔子...
```

**Firmware 只需燒一次**，切換語言只需重燒 model 分區（30 秒）：

```bash
# 換英文
esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write-flash 0x1A0000 firmware/model/model.bin

# 換中文
esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write-flash 0x1A0000 firmware/model_zh/model_zh.bin
```

### 隨機度調整

按 **`t`** 循環切換溫度：

| 溫度 | 效果 |
|------|------|
| 0.1 | 幾乎固定（如同 greedy） |
| 0.5 | 小幅變化 |
| **0.8** | 適中（預設） |
| 1.0 | 變化多 |
| 1.2~1.5 | 很隨機，可能不連貫 |

### 自訂 Prompt（透過電腦）

```bash
source .venv/bin/activate
python src/interact.py --lang zh

# 或英文模式
python src/interact.py --lang en
```

輸入文字後按 Enter，會自動 tokenize 並送給 CoreS3 生成故事。

### 自行編譯

#### 英文模型

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 下載資料集 + 訓練 tokenizer
python data/prepare.py --vocab 32768

# 訓練模型（可選，約 4 小時 CPU）
python src/train.py --arm ple --vocab 32768 --d-model 96 --n-layers 6 \
  --ple-dim 128 --target-core 560000 --batch-size 8 --seq-len 256 \
  --steps 5000 --seed 0 --tag cleandeploy

# 匯出 + 驗證
python src/export.py ple-cleandeploy-s0
cc -O3 -o /tmp/esp32-llm-verify firmware/host_verify/verify.c -lm
/tmp/esp32-llm-verify firmware/model/model.bin firmware/model/golden.txt
```

#### 中文模型

```bash
# 下載中文 TinyStories 資料集
python -c "
from datasets import load_dataset
import json
ds = load_dataset('adam89/TinyStoriesChinese', split='train', streaming=True)
with open('data/tinystories_zh.txt', 'w') as f:
    for row in ds:
        text = json.loads(row['jsonl'])['story_zh'].strip()
        if text: f.write(text + '<|endoftext|>\n')
"

# 訓練中文 tokenizer + 編碼
python -c "
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train(['data/tinystories_zh.txt'], trainers.BpeTrainer(vocab_size=32768, special_tokens=['<|endoftext|>']))
tok.save('data/bpe32768_zh.json')
"

# 訓練 + 匯出中文模型（同英文架構，使用中文資料）
python src/train.py --arm ple --vocab 32768 --d-model 96 --n-layers 6 \
  --ple-dim 128 --target-core 560000 --batch-size 8 --seq-len 256 \
  --steps 5000 --seed 0 --tag zh

python src/export.py ple-zh-s0
```

#### 編譯 Firmware

```bash
# 產生 vocab.h（英文）
python src/gen_assets.py

# 產生 vocab_zh.h（中文）
python -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('data/bpe32768_zh.json')
V = tok.get_vocab_size(); blob = bytearray(); offs = [0]
for i in range(V):
    s = tok.decode([i]); blob.extend(s.encode('utf-8')); offs.append(len(blob))
with open('firmware/zh/vocab_zh.h', 'w') as f:
    f.write('#define VOCAB_N_ZH ' + str(V) + '\n')
    f.write('static const unsigned char VOCAB_BLOB_ZH[' + str(len(blob)) + '] = {' + ','.join(str(b) for b in blob) + '};\n')
    f.write('static const int VOCAB_OFF_ZH[' + str(len(offs)) + '] = {' + ','.join(str(o) for o in offs) + '};\n')
"

# 編譯雙語 Firmware
arduino-cli compile \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-dual-build \
  firmware/esp32_llm_dual
```

### 專案結構

```
├── firmware/
│   ├── esp32_llm/           # 英文版 firmware（獨立）
│   ├── esp32_llm_zh/        # 中文版 firmware（獨立）
│   ├── esp32_llm_dual/      # 雙語 firmware（推薦，中英切換）
│   │   ├── esp32_llm_dual.ino
│   │   ├── display.h        # CoreS3 LCD（含中文字型）
│   │   └── partitions.csv   # app=2MB, model=0x1A0000
│   ├── model/               # 英文模型二進位檔
│   ├── model_zh/            # 中文模型二進位檔
│   ├── zh/vocab_zh.h        # 中文詞彙表
│   └── common/llm.h         # C 推論引擎
├── src/                     # Python 訓練與工具
│   ├── train.py             # 訓練腳本
│   ├── export.py            # 匯出 model.bin
│   ├── gen_assets.py        # 產生 vocab.h
│   └── interact.py          # 互動提示工具（--lang zh/en）
├── data/
│   ├── bpe32768.json        # 英文 tokenizer
│   └── bpe32768_zh.json     # 中文 tokenizer
└── experiments/
```

### 版本歷史

| Tag | 說明 |
|-----|------|
| v1.0 | CoreS3 serial-only 支援 |
| v2.0 | 中文模型 + firmware |
| v2.1 | 雙語 firmware（按 `l` 切換） |
| v2.2 | LCD 中文字型（efontCN） |
| **v2.3** | **隨機取樣（按 `t` 調整溫度）** |

### 鳴謝

- 原始專案：[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
- Per-Layer Embeddings 來自 Google Gemma
- 中文資料集：[TinyStoriesChinese](https://huggingface.co/datasets/adam89/TinyStoriesChinese)

---

## <a id="english"></a>English

### Overview

A 28.9M parameter language model running **fully offline** on the **M5Stack CoreS3**, supporting both **English** and **Chinese** story generation. Uses Google's **Per-Layer Embeddings (PLE)** to store 25M parameters in Flash.

| | |
|---|---|
| Parameters | 28.9M total (25M in Flash lookup table) |
| Chip | ESP32-S3, 512KB SRAM + 8MB PSRAM + 16MB Flash |
| Speed | ~9.5 tok/s |
| Model size | 14.9MB (4-bit quantized) |
| Training data | TinyStories (EN) / TinyStoriesChinese (ZH) |

### Features

- ✅ **Bilingual** — Single firmware with EN/ZH vocab, press `l` to switch
- ✅ **Chinese LCD display** — M5GFX efontCN_16 for full CJK rendering
- ✅ **Temperature sampling** — Press `t` to adjust randomness (0.1–1.5)
- ✅ 10 preset prompts per language
- ✅ Custom token ID input (advanced)
- ✅ Trained entirely on CPU (~4 hours)

### Quick Start

```bash
# Erase flash (required for first-time or partition change)
esptool --chip esp32s3 --port /dev/ttyACM0 erase-flash

# Flash bilingual firmware (one-time only)
arduino-cli upload -p /dev/ttyACM0 \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-dual-build \
  firmware/esp32_llm_dual

# Flash Chinese model (offset 0x1A0000)
esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write-flash 0x1A0000 firmware/model_zh/model_zh.bin

# Monitor
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

### Switching Language

At the serial menu, press **`l`** to toggle:

```
===== PLE TinyLM =====
Select a prompt (0-9) or:
  l: Switch language
  temp=0.8 > l
语言: 中文
```

**Firmware only needs flashing once.** To switch language, just reflash the model (30s):

```bash
# English model
esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write-flash 0x1A0000 firmware/model/model.bin

# Chinese model
esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write-flash 0x1A0000 firmware/model_zh/model_zh.bin
```

### Temperature Sampling

Press **`t`** to cycle temperature:

| Temp | Effect |
|------|--------|
| 0.1 | Almost deterministic |
| 0.5 | Slight variation |
| **0.8** | Balanced (default) |
| 1.0 | More random |
| 1.2–1.5 | Very random, may be incoherent |

### Interactive Prompt

```bash
python src/interact.py --lang en    # English mode
python src/interact.py --lang zh    # 中文模式
```

### Version History

| Tag | Description |
|-----|------------|
| v1.0 | CoreS3 serial-only support |
| v2.0 | Chinese model & firmware |
| v2.1 | Bilingual firmware (`l` to switch) |
| v2.2 | Chinese LCD font (efontCN) |
| **v2.3** | **Temperature sampling (`t` to adjust)** |

### Credits

- Original project: [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
- Per-Layer Embeddings from Google Gemma
- Chinese dataset: [TinyStoriesChinese](https://huggingface.co/datasets/adam89/TinyStoriesChinese)

---

## License

MIT — see [LICENSE](LICENSE)
