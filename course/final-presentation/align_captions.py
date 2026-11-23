"""Align the printed transcript to the actual WAVs for two-line video captions."""
import argparse
import json
import re
import textwrap
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
import whisper
from whisper.timing import find_alignment
from whisper.tokenizer import get_tokenizer

from render_video import sha256


def phrases(text):
    """Balance caption lengths, keeping sentence boundaries and units together."""
    words = text.split()
    best = {len(words): (0.0, [])}
    for start in reversed(range(len(words))):
        options = []
        for stop in range(start + 1, len(words) + 1):
            phrase = " ".join(words[start:stop])
            if len(textwrap.wrap(phrase, width=42, break_on_hyphens=False)) > 2:
                break
            # A modest fixed cost avoids a rapid succession of tiny captions.
            # Look across the entire passage so the last few words are not orphaned.
            cost = 4 + ((len(phrase) - 62) / 25) ** 2
            if stop < len(words):
                last = words[stop - 1].lower()
                following = words[stop].lower().rstrip(".,;:?!")
                if last.endswith((".", "?", "!")):
                    cost -= 4
                elif last.endswith((",", ";", ":")):
                    cost -= 2
                if last in {"the", "an", "a", "and", "of", "to", "for", "or", "with"}:
                    cost += 4
                if following in {"percent", "percentage", "million", "thousand", "hundred"}:
                    cost += 6
            if stop - start < 3 and len(words) > 3:
                cost += 8
            tail_cost, tail = best[stop]
            options.append((cost + tail_cost, [phrase] + tail))
        best[start] = min(options, key=lambda choice: choice[0])
    return best[0][1]


def read_wave(wav):
    sr, samples = wavfile.read(wav)
    if samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError("Caption alignment expects mono 16-bit WAV input.")
    factor = gcd(sr, whisper.audio.SAMPLE_RATE)
    return resample_poly(samples.astype("float32") / 32768, whisper.audio.SAMPLE_RATE // factor, sr // factor)


def cut_times(model, tokenizer, audio, text, pieces):
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    # Whisper's teacher-forced attention aligns the known display text. The
    # script, including numerical formatting, stays verbatim in the captions.
    mel = whisper.log_mel_spectrogram(audio)
    raw = find_alignment(model, tokenizer, tokenizer.encode(" " + text),
                         whisper.pad_or_trim(mel, whisper.audio.N_FRAMES), mel.shape[-1])
    token_starts = {}
    offset = 0
    for word in raw:
        if word.word.strip():
            token_starts[offset] = float(word.start)
            offset += len(re.sub(r"\s+", "", word.word))
    if offset != len(re.sub(r"\s+", "", text)):
        raise ValueError(f"Alignment lost part of the transcript: {text}")
    cut = 0
    starts = [0.0]
    for part in pieces[:-1]:
        cut += len(re.sub(r"\s+", "", part))
        starts.append(round(token_starts[cut], 3))
    ends = starts[1:] + [duration]
    if any(end <= start for start, end in zip(starts, ends)):
        raise ValueError(f"Non-positive caption duration: {text}")
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local Whisper tiny.en checkpoint")
    parser.add_argument("--output", type=Path, default=Path("caption-timings.json"))
    args = parser.parse_args()
    recordings = json.loads(args.recordings.read_text())
    torch.set_num_threads(3)
    model = whisper.load_model(str(args.model), device="cpu")
    tokenizer = get_tokenizer(model.is_multilingual, language="en", task="transcribe")
    aligned = {}
    for clip in recordings["clips"]:
        wav = args.recordings.parent / clip["file"]
        if sha256(wav) != clip["sha256"]:
            raise ValueError(f"Changed audio recording: {wav.name}")
        text = clip["text"]
        pieces = phrases(text)
        starts = (cut_times(model, tokenizer, read_wave(wav), text, pieces)
                  if len(pieces) > 1 else [0.0])
        aligned[clip["sha256"]] = [{"start": start, "text": part} for start, part in zip(starts, pieces)]
    args.output.write_text(json.dumps({"method": "Whisper forced alignment of the exact display text",
        "model_sha256": sha256(args.model), "whisper_version": whisper.__version__,
        "clips": aligned}, indent=2) + "\n")
    print(f"Aligned {sum(map(len, aligned.values()))} captions to {len(aligned)} passages.")


if __name__ == "__main__":
    main()
