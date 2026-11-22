"""实验一：前处理消融实验脚本（Demucs 去伴奏前后 DER / WER 对比）。

对照设置：
- Baseline：直接输入带背景音/音乐的原视频音频
- Ours    ：先经 Demucs (htdemucs) 分离提取 vocals.wav 后再输入 WhisperX + pyannote 3.1

输出指标：
- Diarization Error Rate (DER, %)
- Word Error Rate / Character Error Rate (WER/CER, %)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 Demucs 去伴奏对说话人分离与识别准确率的影响")
    parser.add_argument("--dataset-dir", type=Path, default=Path("experiments/data"), help="含真值 RTTM/SRT 的测试集目录")
    parser.add_argument("--output-json", type=Path, default=Path("experiments/results_diarization.json"), help="实验结果输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (成员 A): 加载测试样本与真值标注，分别统计 raw vs demucs 的 DER 与 WER 并导出 JSON/图表
    print(f"[eval_diarization] 待接入实验数据目录: {args.dataset_dir} -> 输出: {args.output_json}")


if __name__ == "__main__":
    main()
