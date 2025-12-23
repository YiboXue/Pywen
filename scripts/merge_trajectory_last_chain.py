#!/usr/bin/env python3
"""
Merge the last LLM interaction's response into its input_messages as an assistant message.

Usage:
  # 单个文件
  python scripts/merge_trajectory_last_chain.py -i /home/xybwork/.pywen/trajectories/trajectory_20251223_122037.json
  python scripts/merge_trajectory_last_chain.py -i trajectory.json -o merged.json

  # 批量：转换目录下所有 .json（不递归）
  python scripts/merge_trajectory_last_chain.py --input-dir /home/xybwork/.pywen/trajectories --output-dir ./trajectory

说明：
- 单文件模式：未指定 -o 时打印到 stdout。
- 目录模式：输出目录未指定时，默认在当前工作目录下创建 trajectory/ 并写入同名文件。
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def merge_last_interaction(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    interactions = data.get("llm_interactions") or []
    if not interactions:
        raise ValueError("No llm_interactions found in trajectory.")

    last = interactions[-1]
    messages = list(last.get("input_messages") or [])
    resp = last.get("response") or {}

    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
    if resp.get("tool_calls"):
        assistant_msg["tool_calls"] = resp["tool_calls"]

    messages.append(assistant_msg)
    return messages


def main():
    parser = argparse.ArgumentParser(description="Merge last llm_interaction response into input_messages.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Path to a single trajectory json.")
    group.add_argument("--input-dir", help="Path to a directory containing trajectory json files (non-recursive).")
    parser.add_argument("-o", "--output", help="Optional output path for single file mode; if omitted, prints to stdout.")
    parser.add_argument("--output-dir", help="Output directory for batch mode; default: ./trajectory")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        merged = merge_last_interaction(input_path)

        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Merged chain written to: {out_path}")
        else:
            print(json.dumps(merged, ensure_ascii=False, indent=2))
        return

    # Batch mode
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir or "./trajectory").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(in_dir.glob("*.json")):
        try:
            merged = merge_last_interaction(path)
        except Exception as e:
            print(f"⚠️ Skip {path.name}: {e}")
            continue
        out_path = out_dir / path.name
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    print(f"✅ Batch merged {count} file(s) to: {out_dir}")


if __name__ == "__main__":
    main()
