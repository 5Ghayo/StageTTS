"""Read subtitle boundaries from text tracks or burned-in video frames."""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TEXT_SUBTITLE_CODECS = {"ass", "ssa", "subrip", "webvtt", "mov_text", "text", "ttml"}
SRT_TIMING = re.compile(r"^(\d+:\d{2}:\d{2},\d{3})\s+-->\s+(\d+:\d{2}:\d{2},\d{3})(?:\s.*)?$", re.M)


def srt_cut_seconds(subtitles: str) -> list[float]:
    def seconds(stamp: str) -> float:
        hours, minutes, tail = stamp.split(":")
        seconds, millis = tail.split(",")
        return 3600 * int(hours) + 60 * int(minutes) + int(seconds) + int(millis) / 1000

    cuts = set()
    for start, end in SRT_TIMING.findall(subtitles.replace("\r\n", "\n")):
        start_sec, end_sec = seconds(start), seconds(end)
        if end_sec > start_sec:
            cuts.update((start_sec, end_sec))
    return sorted(cuts)


def detect_subtitle_boundaries(source: Path, *, videosubfinder_cli: str | None = None) -> dict | None:
    probe = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "stream=index,codec_name,codec_type:stream_disposition=default,forced,attached_pic:stream_tags=language",
         "-of", "json", str(source)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    candidates = [
        stream for stream in streams
        if stream.get("codec_type") == "subtitle" and stream.get("codec_name") in TEXT_SUBTITLE_CODECS
    ]
    candidates.sort(key=lambda stream: (
        stream.get("disposition", {}).get("forced", 0),
        -stream.get("disposition", {}).get("default", 0),
        stream["index"],
    ))
    for stream in candidates:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", f"0:{stream['index']}",
             "-c:s", "srt", "-f", "srt", "pipe:1"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode:
            continue
        cuts = srt_cut_seconds(result.stdout)
        if cuts:
            return {
                "source": str(source), "stream": stream["index"], "codec": stream["codec_name"],
                "language": stream.get("tags", {}).get("language"), "cut_seconds": cuts,
            }
    if not any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams):
        return None
    executable = videosubfinder_cli or shutil.which("VideoSubFinderCli.run") or shutil.which("VideoSubFinderCli")
    if executable is None:
        return None
    with tempfile.TemporaryDirectory(prefix="stagetts_videosubfinder_") as temp:
        timing = Path(temp) / "subtitles.srt"
        print(f"Scanning video for burned-in subtitles: {source.name}", flush=True)
        subprocess.run(
            [executable, "-r", "-i", str(source.resolve()), "-o", temp, "-ces", str(timing),
             "-dsi", "-te", "0.22", "-nthr", "4"],
            check=True,
        )
        if not timing.is_file():
            return None
        cuts = srt_cut_seconds(timing.read_text(encoding="utf-8-sig"))
    if not cuts:
        return None
    return {"source": str(source), "detector": "videosubfinder", "cut_seconds": cuts}
