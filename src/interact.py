#!/usr/bin/env python3
"""連線 CoreS3 序列埠，將輸入的文字用 BPE tokenizer 轉成 token IDs
送給裝置，讓它根據你的 prompt 生成故事。

使用方式：
  python src/interact.py                    # 自動偵測 /dev/ttyACM0
  python src/interact.py --port /dev/ttyACM0
"""

import argparse
import os
import time

import serial
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_PATH = os.path.join(HERE, "..", "data", "bpe32768.json")


def main():
    ap = argparse.ArgumentParser(description="ESP32-AI 互動提示工具")
    ap.add_argument("--port", "-p", default="/dev/ttyACM0", help="序列埠 (預設 /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200, help="鮑率 (預設 115200)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOK_PATH)
    ser = serial.Serial(args.port, args.baud, timeout=10)
    time.sleep(2)  # wait for reset

    print(f"已連線 {args.port}")
    print(f"Tokenizer: {TOK_PATH}")
    print("輸入故事開頭後按 Enter，留空離開。\n")

    try:
        while True:
            text = input("故事開頭> ").strip()
            if not text:
                break

            ids = tok.encode(text).ids
            if len(ids) > 256:
                print(f"token 太多 ({len(ids)})，上限 256，截斷中")
                ids = ids[:256]

            line = " ".join(str(i) for i in ids) + "\n"
            ser.write(line.encode())
            print(f"  tokens: {len(ids)}  輸入: {text!r}")
            print(f"  還原: {tok.decode(ids)!r}\n")

            # Read and print device output
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
        print("\n離開。")


if __name__ == "__main__":
    main()
