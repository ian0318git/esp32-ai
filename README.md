# ESP32-AI — 28.9M 參數語言模型在 ESP32-S3 上運行

[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v6.0-blue)](https://github.com/espressif/esp-idf)
[![Target](https://img.shields.io/badge/target-ESP32--S3-orange)](https://www.espressif.com/en/products/socs/esp32-s3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🇹🇼 繁體中文** · [English](#english)

---

## 🇹🇼 繁體中文

### 概述

這是一個 **28.9M 參數**的語言模型，能在 **ESP32-S3** 微控制器（約 $8）上**完全離線**運行。採用 Google Gemma 的 **Per-Layer Embeddings (PLE)** 技術，將 25M 參數的大表放進 Flash，每 token 僅讀取約 450 bytes，讓記憶體受限的 MCU 也能跑大型語言模型。

**核心數據：**
| | |
|---|---|
| 參數量 | 28.9M（其中 25M 在 Flash 查找表） |
| 晶片 | ESP32-S3，512KB SRAM + 8MB PSRAM + 16MB Flash |
| 速度 | ~9.5 tok/s |
| 模型大小 | 14.9MB（4-bit 量化） |
| 訓練資料 | TinyStories |

### 本專案修改

在原專案 [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) 基礎上，新增 **M5Stack CoreS3** 支援：

- ✅ 內建 **ILI9342C 彩色 LCD** 顯示（320x240，使用 M5GFX 驅動）
- ✅ CoreS3 專用板子設定（QSPI PSRAM、USB CDC）
- ✅ 已訓練模型權重（val ppl 13.9，5000 steps，batch=8）
- ✅ 可直接燒錄使用

### 硬體需求

- **M5Stack CoreS3**（或其他 ESP32-S3 N16R8 開發板）
- USB-C 連接線

### 快速開始

```bash
# 1. 燒錄 Firmware
arduino-cli upload \
  -p /dev/ttyACM0 \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-build-cores3 \
  firmware/esp32_llm

# 2. 燒錄模型到 Flash 分區
pip install esptool
esptool.py --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write_flash 0x110000 firmware/model/model.bin

# 3. 監控序列輸出
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

### 自行編譯

```bash
# Python 環境
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 下載資料集 + 訓練 tokenizer
python data/prepare.py --vocab 32768

# 訓練模型（可選，約 4 小時在 CPU / 更快在 GPU）
python src/train.py --arm ple --vocab 32768 --d-model 96 --n-layers 6 \
  --ple-dim 128 --target-core 560000 --batch-size 8 --seq-len 256 \
  --steps 5000 --seed 0 --tag cleandeploy

# 匯出模型二進位檔
python src/export.py ple-cleandeploy-s0

# 產生 vocab.h
python src/gen_assets.py

# 驗證 C 語言推論
cc -O3 -o /tmp/esp32-llm-verify firmware/host_verify/verify.c -lm
/tmp/esp32-llm-verify firmware/model/model.bin firmware/model/golden.txt

# 編譯 Firmware（CoreS3）
arduino-cli compile \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build-cores3 \
  firmware/esp32_llm
```

### 專案結構

```
├── firmware/
│   ├── esp32_llm/          # Arduino 韌體（CoreS3）
│   │   ├── esp32_llm.ino   # 主程式
│   │   ├── display.h       # 顯示驅動（OLED / TFT / CoreS3）
│   │   ├── partitions.csv  # 分區表
│   │   └── vocab.h         # 詞彙表（自動產生）
│   ├── common/llm.h        # C 語言推論引擎
│   ├── host_verify/        # Host 端驗證
│   └── model/              # 模型二進位檔
├── src/                    # Python 訓練與工具
├── data/                   # 資料集與 tokenizer
└── experiments/            # 實驗腳本
```

### 鳴謝

- 原始專案：[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) — 感謝 slvDev 的開創性工作
- Per-Layer Embeddings 技術來自 Google Gemma
- TinyStories 資料集來自 Microsoft Research

---

## <a id="english"></a>English

### Overview

This is a **28.9M parameter** language model that runs **fully offline** on an **ESP32-S3** microcontroller (~$8). It uses Google's **Per-Layer Embeddings (PLE)** technique to store 25M parameters in Flash, reading only ~450 bytes per token, making large language models feasible on memory-constrained MCUs.

**Key numbers:**

| | |
|---|---|
| Parameters | 28.9M total (25M in Flash lookup table) |
| Chip | ESP32-S3, 512KB SRAM + 8MB PSRAM + 16MB Flash |
| Speed | ~9.5 tok/s |
| Model size | 14.9MB (4-bit quantized) |
| Training data | TinyStories |

### This Fork

Based on [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai), this fork adds **M5Stack CoreS3** support:

- ✅ Built-in **ILI9342C color LCD** display (320x240 via M5GFX driver)
- ✅ CoreS3 board configuration (QSPI PSRAM, USB CDC)
- ✅ Pre-trained model weights (val ppl 13.9, 5000 steps, batch=8)
- ✅ Ready to flash and run

### Hardware Required

- **M5Stack CoreS3** (or any ESP32-S3 N16R8 board)
- USB-C cable

### Quick Start

```bash
# 1. Flash firmware
arduino-cli upload \
  -p /dev/ttyACM0 \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-build-cores3 \
  firmware/esp32_llm

# 2. Flash model to partition
pip install esptool
esptool.py --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write_flash 0x110000 firmware/model/model.bin

# 3. Monitor serial output
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

### Building from Source

```bash
# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Download dataset + train tokenizer
python data/prepare.py --vocab 32768

# Train model (optional, ~4 hours on CPU / faster on GPU)
python src/train.py --arm ple --vocab 32768 --d-model 96 --n-layers 6 \
  --ple-dim 128 --target-core 560000 --batch-size 8 --seq-len 256 \
  --steps 5000 --seed 0 --tag cleandeploy

# Export model binary
python src/export.py ple-cleandeploy-s0

# Generate vocab.h
python src/gen_assets.py

# Verify C inference
cc -O3 -o /tmp/esp32-llm-verify firmware/host_verify/verify.c -lm
/tmp/esp32-llm-verify firmware/model/model.bin firmware/model/golden.txt

# Compile firmware (CoreS3)
arduino-cli compile \
  --fqbn 'esp32:esp32:m5stack_cores3:PSRAM=enabled,UploadSpeed=921600,PartitionScheme=custom,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build-cores3 \
  firmware/esp32_llm
```

### Project Structure

```
├── firmware/
│   ├── esp32_llm/          # Arduino firmware (CoreS3)
│   │   ├── esp32_llm.ino   # Main sketch
│   │   ├── display.h       # Display drivers (OLED / TFT / CoreS3)
│   │   ├── partitions.csv  # Flash partition table
│   │   └── vocab.h         # Vocabulary (generated)
│   ├── common/llm.h        # C inference engine
│   ├── host_verify/        # Host verification tools
│   └── model/              # Model binary
├── src/                    # Python training & tools
├── data/                   # Dataset & tokenizer
└── experiments/            # Experiment scripts
```

### Credits

- Original project: [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) — thanks to slvDev for the pioneering work
- Per-Layer Embeddings from Google Gemma
- TinyStories dataset from Microsoft Research

---

## License

MIT — see [LICENSE](LICENSE)
