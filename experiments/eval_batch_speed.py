"""实验二：按角色分组批量合成 vs 逐句切换权重 耗时与显存监控对比脚本。

对照设置：
- Baseline（逐句顺序合成）：按时间顺序遍历 30 句双人对话，频繁触发 GPT/SoVITS 权重热切换
- Ours（按 speaker 分组批处理）：先按说话人聚类再批量合成，每角色仅切 1 次权重

输出指标：
- 总合成耗时（秒）与平均单句耗时
- 推理期间 GPU 显存峰值与波动曲线（MB，建议通过 pynvml / nvidia-smi 采样）
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测按说话人分组批处理与逐句切换权重的耗时及显存表现")
    parser.add_argument("--script-path", type=Path, default=Path("workspace/proj_demo_01/script.json"), help="测试剧本路径")
    parser.add_argument("--output-json", type=Path, default=Path("experiments/results_batch_speed.json"), help="实验结果输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (成员 A & B): 调用 TTS API 执行两组模式耗时与显存采样记录
    print(f"[eval_batch_speed] 待运行对比评测，剧本: {args.script_path} -> 输出: {args.output_json}")


if __name__ == "__main__":
    main()
