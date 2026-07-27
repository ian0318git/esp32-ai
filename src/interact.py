#!/usr/bin/env python3
"""Interactive prompt tool for ESP32-AI.

Connects to the CoreS3 over serial, reads user text, tokenizes it with the
matching BPE tokenizer, and sends token IDs to the device for generation.

Usage:
  python src/interact.py                    # auto-detect /dev/ttyACM0
  python src/interact.py --port /dev/ttyACM0
"""

import argparse
import sys
import time

import serial
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_PATH = os.path.join(HERE, "..", "data", "bpe32768.json")


def main():
    ap = argparse.ArgumentParser(description="Interactive prompt for ESP32-AI")
    ap.add_argument("--port", "-p", default="/dev/ttyACM0", help="Serial port")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate")
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOK_PATH)
    ser = serial.Serial(args.port, args.baud, timeout=10)
    time.sleep(2)  # wait for reset

    print(f"Connected to {args.port}")
    print(f"Tokenizer: {TOK_PATH}")
    print("Type your prompt and press Enter. Empty line exits.\n")

    try:
        while True:
            text = input("prompt> ").strip()
            if not text:
                break

            ids = tok.encode(text).ids
            if len(ids) > 256:
                print(f"Too many tokens ({len(ids)}), max 256. Truncating.")
                ids = ids[:256]

            # Send token IDs as space-separated text
            line = " ".join(str(i) for i in ids) + "\n"
            ser.write(line.encode())
            print(f"  tokens: {len(ids)}  sent: {text!r}")
            print(f"  round-trip: {tok.decode(ids)!r}\n")

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
        print("\nDone.")


if __name__ == "__main__":
    main()
