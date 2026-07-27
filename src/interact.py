#!/usr/bin/env python3
"""Connect to CoreS3 serial, tokenize your prompt with BPE tokenizer, send
token IDs to the device, and print the generated story.

Usage:
  python src/interact.py                    # 中文模式 (default)
  python src/interact.py --lang en          # English mode
  python src/interact.py --port /dev/ttyACM0
"""

import argparse
import os
import time

import serial
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_PATH = os.path.join(HERE, "..", "data", "bpe32768.json")

LANG = {
    "zh": {
        "title": "ESP32-AI 互動故事工具",
        "prompt": "故事開頭> ",
        "connected": "已連線 {}",
        "tok_path": "Tokenizer: {}",
        "hint": "輸入故事開頭後按 Enter，留空離開。\n",
        "too_many": "token 太多 ({})，上限 256，截斷中",
        "tokens": "  tokens: {}  輸入: {!r}",
        "roundtrip": "  還原: {!r}\n",
        "goodbye": "\n離開。",
        "port_help": "序列埠 (預設 /dev/ttyACM0)",
        "baud_help": "鮑率 (預設 115200)",
        "lang_help": "語言: zh=中文, en=English",
    },
    "en": {
        "title": "ESP32-AI Interactive Prompt Tool",
        "prompt": "prompt> ",
        "connected": "Connected to {}",
        "tok_path": "Tokenizer: {}",
        "hint": "Type your story prompt and press Enter. Empty line exits.\n",
        "too_many": "Too many tokens ({}), max 256. Truncating.",
        "tokens": "  tokens: {}  input: {!r}",
        "roundtrip": "  round-trip: {!r}\n",
        "goodbye": "\nDone.",
        "port_help": "Serial port (default /dev/ttyACM0)",
        "baud_help": "Baud rate (default 115200)",
        "lang_help": "Language: zh=中文, en=English",
    },
}


def main():
    ap = argparse.ArgumentParser(description="ESP32-AI Interactive Prompt Tool")
    ap.add_argument("--port", "-p", default="/dev/ttyACM0", help="...")
    ap.add_argument("--baud", type=int, default=115200, help="...")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="...")
    args = ap.parse_args()
    T = LANG[args.lang]

    # Update help text
    ap._actions[1].help = T["port_help"]
    ap._actions[2].help = T["baud_help"]
    ap._actions[3].help = T["lang_help"]
    ap.description = T["title"]

    tok = Tokenizer.from_file(TOK_PATH)
    ser = serial.Serial(args.port, args.baud, timeout=10)
    time.sleep(2)

    print(T["connected"].format(args.port))
    print(T["tok_path"].format(TOK_PATH))
    print(T["hint"])

    try:
        while True:
            text = input(T["prompt"]).strip()
            if not text:
                break

            ids = tok.encode(text).ids
            if len(ids) > 256:
                print(T["too_many"].format(len(ids)))
                ids = ids[:256]

            line = " ".join(str(i) for i in ids) + "\n"
            ser.write(line.encode())
            print(T["tokens"].format(len(ids), text))
            print(T["roundtrip"].format(tok.decode(ids)))

            start = time.time()
            while time.time() - start < 60:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
                    print(data, end="", flush=True)
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(T["goodbye"])


if __name__ == "__main__":
    main()
