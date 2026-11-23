"""Apply the approved, fixed, pitch-preserving tempo to the three voice tracks."""
import argparse
import json
from pathlib import Path
import subprocess

import soundfile as sf

from render_video import find_ffmpeg, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tempo", type=float, default=1.45)
    args = parser.parse_args()
    if not 0.5 <= args.tempo <= 2:
        parser.error("Tempo must be between 0.5 and 2.")
    raw = json.loads(args.recordings.read_text())
    if raw["status"] != "complete" or len(raw["clips"]) != raw["expected_clips"]:
        raise ValueError("Source recording set is incomplete.")
    ffmpeg, version = find_ffmpeg()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    clips = []
    for row in raw["clips"]:
        source = args.recordings.parent / row["file"]
        if sha256(source) != row["sha256"]:
            raise ValueError(f"Recording changed: {source.name}")
        filename = f"slide-{row['slide']:02d}-{row['index']:02d}.wav"
        dest = args.output_dir / filename
        temp = dest.with_suffix(".tmp.wav")
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                        "-af", f"apad=pad_dur=0.06,atempo={args.tempo}", "-c:a", "pcm_s16le",
                        "-ar", "24000", "-ac", "1", str(temp)], check=True)
        info = sf.info(temp)
        if info.frames == 0 or abs(info.duration - row["duration_seconds"] / args.tempo) > 0.22:
            raise ValueError(f"Unexpected duration change: {filename}")
        temp.replace(dest)
        clips.append({"slide": row["slide"], "index": row["index"], "text": row["text"],
                      "spoken_text": row["synthesis_text"], "voice": row["voice"],
                      "file": filename, "sha256": sha256(dest), "source_sha256": row["sha256"],
                      "duration_seconds": info.duration, "source_seconds": row["duration_seconds"]})
    plan = raw["speaker_plan"]
    for speaker in plan["speakers"].values():
        speaker["speed"] = args.tempo
    meta = raw["engine_metadata"].copy()
    meta["source_recording_postprocessing"] = meta.pop("postprocessing", None)
    meta.update({"postprocessing": "60 ms of protective trailing silence, then a uniform pitch-preserving FFmpeg atempo filter on each complete passage.",
                 "requested_playback_tempo": args.tempo, "tempo_ffmpeg_version": version,
                 "pace_script_sha256": sha256(__file__), "passage_source_sha256": raw["input_sha256"],
                 "raw_speech_seconds": sum(c["source_seconds"] for c in clips),
                 "paced_speech_seconds": sum(c["duration_seconds"] for c in clips)})
    bundle = {"speaker_plan": plan, "engine_metadata": meta, "clips": clips}
    (args.output_dir / "recordings.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"{len(clips)} passages. Source {meta['raw_speech_seconds']:.2f}s, paced {meta['paced_speech_seconds']:.2f}s.")


if __name__ == "__main__":
    main()
