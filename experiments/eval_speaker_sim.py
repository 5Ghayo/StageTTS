"""实验四：动态情绪参考音频路由 vs 静态单参考音 声纹相似度与客观声学评测脚本。

对照设置：
- Baseline：全角色仅绑定单段 neutral.wav 静态参考音
- Ours    ：基于情绪标签动态路由 angry/happy/sad/whisper/neutral 对应参考音频

输出指标：
- 基于 ECAPA-TDNN 提取说话人 Embedding 计算的余弦相似度（Cosine Similarity）
- 结合人工盲听表计算自然度与情绪贴合度 MOS（Mean Opinion Score）
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 ECAPA-TDNN 计算合成音频与参考音频的声纹余弦相似度")
    parser.add_argument("--script-path", type=Path, default=Path("workspace/proj_demo_01/script.json"), help="待评测项目 script.json")
    parser.add_argument("--output-json", type=Path, default=Path("experiments/results_speaker_sim.json"), help="评测结果输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (成员 A): 加载 SpeechBrain / ModelScope ECAPA-TDNN 提取特征并汇总各情绪相似度均值
    print(f"[eval_speaker_sim] 待运行声纹相似度评测，剧本: {args.script_path} -> 输出: {args.output_json}")


if __name__ == "__main__":
    main()
