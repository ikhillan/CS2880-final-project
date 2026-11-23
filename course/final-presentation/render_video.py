#!/usr/bin/env python3
"""Narrate a Beamer PDF with local Kokoro speech and measured slide timings.

Run with the isolated narration Python, after revising the guide and PDF:
  .build/tts/venv/bin/python render_video.py --check
  .build/tts/venv/bin/python render_video.py --audio-only
  .build/tts/venv/bin/python render_video.py

The guide format is ``SLIDE 1 — Title``, then
``00:00–00:30 | 30 seconds``, then spoken paragraphs. Headings and time cues
are not spoken. Default natural timing follows measured speech and the
specified pauses. Optional ``--timing guide`` distributes spare time into
short sentence pauses and a final hold. Neither mode clips speech or
automatically accelerates it. Overruns against guide targets are reported.

Dependencies: kokoro-onnx, onnxruntime, soundfile, numpy, imageio-ffmpeg;
Poppler's pdftoppm/pdfinfo. No network requests or automatic installs occur.
Kokoro model/voice files are provided separately; see the ignored local
.build/tts/model-provenance.json and the primary sources recorded below.
The voice is an ordinary synthetic voice, not any author's voice.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 24000
MODEL_SOURCE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1'
BASE_MODEL = 'https://huggingface.co/hexgrad/Kokoro-82M'
HEADER = re.compile(r'^SLIDE\s+(\d+)\s*[—–-]\s*(.+)$', re.MULTILINE)
TIMING = re.compile(r'^\s*(\d+):(\d{2})\s*[—–-]\s*(\d+):(\d{2})\s*\|\s*(\d+)\s*seconds?\s*$', re.MULTILINE)


@dataclass
class Slide:
    number: int
    title: str
    text: str
    planned_seconds: float


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command, **kwargs):
    result = subprocess.run([str(x) for x in command], capture_output=True,
                            text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed ({result.returncode}):\n'
                           + result.stderr[-6000:])
    return result.stdout


def read_guide(path, source=None):
    if source is None:
        source = Path(path).read_text()
    headers = list(HEADER.finditer(source))
    if not headers:
        raise ValueError('No SLIDE headings found in presenter guide.')
    slides, expected_start = [], 0
    for n, match in enumerate(headers):
        end = headers[n + 1].start() if n + 1 < len(headers) else len(source)
        block = source[match.end():end]
        timing = TIMING.search(block)
        if not timing:
            raise ValueError(f'Missing timing line for slide {n+1}.')
        start = int(timing[1]) * 60 + int(timing[2])
        finish = int(timing[3]) * 60 + int(timing[4])
        duration = int(timing[5])
        if start != expected_start or finish - start != duration or duration <= 0:
            raise ValueError(f'Non-contiguous or inconsistent timing on slide {n+1}.')
        expected_start = finish
        text = block[timing.end():].strip()
        text = re.sub(r'^\s*\[(?:pause|brief pause)\]\s*$', '', text,
                      flags=re.MULTILINE | re.IGNORECASE).strip()
        if not text or int(match[1]) != n + 1:
            raise ValueError(f'Empty or out-of-order slide {n+1}.')
        if re.search(r'\\(?:frac|sum|begin|text|mathbf)|\$[^$]+\$', text):
            raise ValueError(f'Slide {n+1} contains unspoken TeX; write it in words.')
        slides.append(Slide(n + 1, match[2].strip(), text, duration))
    return slides


def spoken_text(text):
    substitutions = {
        r'\bGPT\b': 'G P T', r'\bDA\b': 'deferred acceptance',
        r'\bNYC\b': 'New York City', r'\bAI\b': 'A I',
        r'\bNP[- ]hard\b': 'N P hard', r'\bNP[- ]complete\b': 'N P complete',
        r'\bpp\b': 'percentage points', r'%': ' percent',
    }
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    return text.replace('−', ' minus ').replace('≤', ' at most ').replace('≥', ' at least ')


def speech_chunks(text, limit=220):
    """Keep sentences intact where practical; each cue has measured audio."""
    chunks = []
    for paragraph in re.split(r'\n\s*\n', text):
        paragraph = ' '.join(paragraph.split())
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9“"\'])', paragraph)
        for sentence in sentences:
            if not sentence:
                continue
            while len(sentence) > limit:
                # Prefer an existing clause boundary to an arbitrary word cut.
                cuts = [m.end() for m in re.finditer(r'[,;:]\s+', sentence[:limit+1])]
                cut = cuts[-1] if cuts and cuts[-1] > limit / 3 else sentence.rfind(' ', 0, limit)
                if cut <= 0:
                    raise ValueError('An unreasonably long token cannot be narrated.')
                chunks.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                chunks.append(sentence)
    return chunks


def find_ffmpeg(explicit=None):
    candidates = [explicit] if explicit else []
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    candidates.append(shutil.which('ffmpeg'))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            version = run([candidate, '-version']).splitlines()[0]
            return str(candidate), version
        except (RuntimeError, OSError):
            continue
    raise RuntimeError('No working ffmpeg. Install imageio-ffmpeg in the isolated environment.')


def pdf_pages(path):
    info = run(['pdfinfo', path])
    match = re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)
    if not match:
        raise RuntimeError('Could not determine PDF page count.')
    return int(match[1])


def timestamp(seconds, vtt=False):
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return f'{hours:02}:{minutes:02}:{secs:02}{"." if vtt else ","}{millis:03}'


def captions(cues, output):
    srt, vtt = [], ['WEBVTT', '']
    for i, cue in enumerate(cues, 1):
        lines = '\n'.join(textwrap.wrap(cue['text'], width=54, break_long_words=False))
        srt += [str(i), f"{timestamp(cue['start'])} --> {timestamp(cue['end'])}", lines, '']
        vtt += [f"{timestamp(cue['start'], True)} --> {timestamp(cue['end'], True)}", lines, '']
    output.with_suffix('.srt').write_text('\n'.join(srt) + '\n')
    output.with_suffix('.vtt').write_text('\n'.join(vtt) + '\n')


def ffquote(path):
    return "'" + str(Path(path).resolve()).replace("'", "'\\''") + "'"


def extra_pauses(chunks, text, available, base_gap):
    """Fill spare time with at most 1s sentence / 2s paragraph pauses."""
    paragraphs = [' '.join(p.split()) for p in re.split(r'\n\s*\n', text)]
    ends = [any(p.endswith(chunk) for p in paragraphs) for chunk in chunks[:-1]]
    weights = [2.0 if end else 1.0 if re.search(r'[.!?][”"\']?$', chunk) else 0.35
               for chunk, end in zip(chunks[:-1], ends)]
    caps = [max(0, round((2.0 if end else 1.0) * SAMPLE_RATE) - base_gap) for end in ends]
    extra = [0] * len(caps)
    while available > 0:
        active = [i for i in range(len(caps)) if extra[i] < caps[i]]
        if not active:
            break
        weight = sum(weights[i] for i in active)
        added = 0
        for i in active:
            amount = min(caps[i] - extra[i], max(1, int(available * weights[i] / weight)))
            amount = min(amount, available - added)
            extra[i] += amount
            added += amount
            if added == available:
                break
        available -= added
    return extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdf', type=Path, default=HERE / 'presentation.pdf')
    parser.add_argument('--guide', type=Path, default=HERE / 'presenter-guide.txt')
    parser.add_argument('--output', type=Path, default=HERE / 'presentation-narrated.mp4')
    parser.add_argument('--workdir', type=Path, default=HERE / '.build/tts/render')
    parser.add_argument('--model', type=Path, default=HERE / '.build/tts/kokoro-v1.0.onnx')
    parser.add_argument('--voices', type=Path, default=HERE / '.build/tts/voices-v1.0.bin')
    parser.add_argument('--voice', default='af_heart')
    parser.add_argument('--speed', type=float, default=0.88, help='Explicit Kokoro speech speed; never adjusted automatically.')
    parser.add_argument('--timing', choices=['guide', 'natural'], default='natural')
    parser.add_argument('--lead-pause', type=float, default=0.35)
    parser.add_argument('--tail-pause', type=float, default=0.75)
    parser.add_argument('--sentence-pause', type=float, default=0.18)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--check', action='store_true', help='Validate guide, PDF and tools without generating audio/video.')
    parser.add_argument('--audio-only', action='store_true', help='Generate narration, captions and actual timing report; skip video.')
    args = parser.parse_args()
    if not 0.65 <= args.speed <= 1.30:
        parser.error('Use a natural explicit speech speed between 0.65 and 1.30.')
    if min(args.lead_pause, args.tail_pause, args.sentence_pause) < 0 or args.fps < 1 or args.threads < 1:
        parser.error('Pauses must be nonnegative; fps and threads must be positive.')
    guide_bytes = args.guide.read_bytes()
    guide_hash = hashlib.sha256(guide_bytes).hexdigest()
    slides = read_guide(args.guide, source=guide_bytes.decode())
    pdf_bytes = args.pdf.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    if not args.audio_only and pdf_pages(args.pdf) != len(slides):
        raise ValueError('Guide slide count differs from PDF page count; rebuild the final deck first.')
    ffmpeg, ffmpeg_version = find_ffmpeg(args.ffmpeg)
    for path in [args.model, args.voices]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing local speech model asset: {path}')
    print(f'{len(slides)} slides; guide target {sum(s.planned_seconds for s in slides):.1f}s; {ffmpeg_version}', flush=True)
    if args.check:
        for slide in slides:
            words = len(slide.text.split())
            print(f'{slide.number:02}: {slide.planned_seconds:.0f}s, {words} words, {words*60/slide.planned_seconds:.0f} words/min — {slide.title}')
        return

    import numpy as np
    import soundfile as sf
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    kokoro = Kokoro.from_session(session, str(args.voices))
    if args.voice not in kokoro.get_voices():
        raise ValueError(f'Voice {args.voice} is absent from the supplied voice pack.')
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.workdir / 'audio-cache'
    cache.mkdir(exist_ok=True)
    model_hash, voices_hash = sha256(args.model), sha256(args.voices)
    engine_version = importlib.metadata.version('kokoro-onnx')
    spoken_frames, cues, records = 0, [], []
    narration_path = args.workdir / 'narration.wav'
    with sf.SoundFile(narration_path, 'w', samplerate=SAMPLE_RATE, channels=1, subtype='PCM_16') as full_audio:
        for slide in slides:
            offset = spoken_frames / SAMPLE_RATE
            chunks, chunk_info, audio_chunks = speech_chunks(slide.text), [], []
            for index, chunk in enumerate(chunks):
                spoken = spoken_text(chunk)
                key_data = {'text': spoken, 'voice': args.voice, 'speed': args.speed,
                            'model': model_hash, 'voices': voices_hash,
                            'engine': engine_version, 'lang': 'en-us', 'sentence_pause': 0.15, 'clause_pause': 0.08}
                key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
                wav = cache / f'{key}.wav'
                if wav.exists():
                    samples, sample_rate = sf.read(wav, dtype='float32')
                else:
                    samples, sample_rate = kokoro.create(spoken, voice=args.voice,
                        speed=args.speed, lang='en-us', sentence_pause=0.15, clause_pause=0.08)
                    sf.write(wav, samples, sample_rate, subtype='PCM_16')
                    samples, sample_rate = sf.read(wav, dtype='float32')
                if sample_rate != SAMPLE_RATE or samples.ndim != 1 or not len(samples):
                    raise ValueError(f'Invalid synthesized audio for slide {slide.number}.')
                if not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 1:
                    raise ValueError(f'Invalid or clipped synthesized audio for slide {slide.number}.')
                audio_chunks.append(samples)
                chunk_info.append({'text': chunk, 'spoken_text': spoken, 'audio_sha256': sha256(wav),
                                   'cache_file': f'audio-cache/{wav.name}'})
            lead_frames = round(args.lead_pause * SAMPLE_RATE)
            gaps = [round(args.sentence_pause * SAMPLE_RATE)] * (len(chunks) - 1)
            natural_frames = lead_frames + sum(len(a) for a in audio_chunks) + sum(gaps) + round(args.tail_pause * SAMPLE_RATE)
            target_frames = round(slide.planned_seconds * SAMPLE_RATE)
            frames = max(natural_frames, target_frames) if args.timing == 'guide' else natural_frames
            extra = frames - natural_frames
            if gaps and extra:
                # Reserve a modest final hold; place other spare time between
                # thoughts. A hard cap prevents very long intra-sentence gaps.
                available = max(0, extra - round(1.25 * SAMPLE_RATE))
                extras = extra_pauses(chunks, slide.text, available, gaps[0])
                gaps = [a+b for a, b in zip(gaps, extras)]
            segments = [np.zeros(lead_frames, dtype=np.float32)]
            cursor = lead_frames
            for index, samples in enumerate(audio_chunks):
                start = offset + cursor / SAMPLE_RATE
                segments.append(samples)
                cursor += len(samples)
                stop = offset + cursor / SAMPLE_RATE
                cues.append({'slide': slide.number, 'text': chunks[index], 'start': start, 'end': stop})
                chunk_info[index].update({'start': start, 'end': stop})
                if index < len(gaps):
                    segments.append(np.zeros(gaps[index], dtype=np.float32))
                    cursor += gaps[index]
            speech_end = cursor / SAMPLE_RATE
            segments.append(np.zeros(frames - cursor, dtype=np.float32))
            audio = np.concatenate(segments)
            assert len(audio) == frames
            full_audio.write(audio)
            actual = frames / SAMPLE_RATE
            overrun = max(0.0, actual - slide.planned_seconds)
            record = {'slide': slide.number, 'title': slide.title, 'start': offset,
                      'end': offset + actual, 'duration': actual,
                      'planned_duration': slide.planned_seconds,
                      'minimum_duration': natural_frames / SAMPLE_RATE,
                      'speech_and_inter_sentence_seconds': speech_end - args.lead_pause,
                      'inter_sentence_pause_seconds': [g / SAMPLE_RATE for g in gaps],
                      'final_hold_seconds': actual - speech_end,
                      'overrun_seconds': overrun, 'words': len(slide.text.split()),
                      'cues': chunk_info}
            records.append(record)
            spoken_frames += frames
            marker = ' OVERRUN' if overrun > 0.25 else ''
            if record['final_hold_seconds'] > 8:
                marker += ' LONG HOLD'
            print(f"Slide {slide.number:02}: {actual:.2f}s / {slide.planned_seconds:.0f}s target; "
                  f"end hold {record['final_hold_seconds']:.2f}s; overrun {overrun:.2f}s{marker}", flush=True)

    if sha256(args.guide) != guide_hash:
        raise RuntimeError('Presenter guide changed during synthesis. Cached speech is retained; rerun against the final guide.')
    previous_end = 0.0
    for slide, record in zip(slides, records):
        if ' '.join(c['text'] for c in record['cues']) != ' '.join(slide.text.split()):
            raise RuntimeError(f'Caption text differs from slide {slide.number} narration.')
        if ' '.join(' '.join(c['spoken_text'].split()) for c in record['cues']) != ' '.join(spoken_text(slide.text).split()):
            raise RuntimeError(f'Spoken normalization differs on slide {slide.number}.')
        for cue in record['cues']:
            if not record['start'] <= cue['start'] < cue['end'] <= record['end']:
                raise RuntimeError(f'Caption extends beyond slide {slide.number}.')
            if cue['start'] < previous_end - 1e-8:
                raise RuntimeError('Caption times overlap or go backwards.')
            previous_end = cue['end']
    captions(cues, args.output)
    total = spoken_frames / SAMPLE_RATE
    metadata_path = args.workdir / 'chapters.ffmetadata'
    metadata = [';FFMETADATA1', 'title=Who gets a place? Recommendations under competition',
                'comment=Narration uses Kokoro af_heart, an ordinary synthetic voice; no author voice cloning.']
    for record in records:
        title = record['title'].replace('\\', '\\\\').replace('=', '\\=').replace(';', '\\;').replace('#', '\\#')
        metadata += ['[CHAPTER]', 'TIMEBASE=1/1000', f"START={round(record['start']*1000)}",
                     f"END={round(record['end']*1000)}", f'title={title}']
    metadata_path.write_text('\n'.join(metadata) + '\n')
    manifest = {'synthetic_narration': True, 'voice': args.voice, 'voice_is_not_an_author': True,
                'engine': 'kokoro-onnx', 'engine_version': engine_version,
                'package_versions': {name: importlib.metadata.version(name)
                                     for name in ['kokoro-onnx', 'onnxruntime', 'soundfile', 'numpy', 'imageio-ffmpeg']},
                'base_model': BASE_MODEL, 'model_release': MODEL_SOURCE,
                'model_license': 'Apache-2.0', 'wrapper_license': 'MIT',
                'model_sha256': model_hash, 'voice_pack_sha256': voices_hash,
                'speech_speed': args.speed, 'automatic_speed_adjustment': False,
                'speech_truncated': False, 'timing_mode': args.timing,
                'guide_sha256': guide_hash, 'pdf_sha256': pdf_hash,
                'renderer_sha256': sha256(__file__),
                'sample_rate': SAMPLE_RATE, 'planned_total_seconds': sum(s.planned_seconds for s in slides),
                'actual_total_seconds': total, 'narration_sha256': sha256(narration_path),
                'captions': 'Measured synthesized sentence/clause boundaries; not forced word alignment.',
                'full_text_and_caption_boundaries_verified': True,
                'ffmpeg_version': ffmpeg_version, 'frames_per_second': args.fps,
                'slides': records, 'video_created': False}
    manifest_path = args.output.parent / 'narration-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Actual narration timeline: {total:.2f}s ({total/60:.2f} min).', flush=True)
    if args.audio_only:
        print(f'Audio-only output: {narration_path.resolve()}', flush=True)
        return

    if sha256(args.pdf) != pdf_hash:
        raise RuntimeError('PDF changed during narration preparation. Rerun against the final PDF; speech remains cached.')
    frozen_pdf = args.workdir / 'render-source.pdf'
    frozen_pdf.write_bytes(pdf_bytes)
    image_dir = args.workdir / 'slides'
    image_dir.mkdir(exist_ok=True)
    pdf_hash_path = image_dir / 'pdf.sha256'
    if not pdf_hash_path.exists() or pdf_hash_path.read_text().strip() != manifest['pdf_sha256']:
        for old_page in image_dir.glob('slide-*.png'):
            old_page.unlink()
        run(['pdftoppm', '-scale-to', '1920', '-png', frozen_pdf, image_dir / 'slide'])
        pdf_hash_path.write_text(manifest['pdf_sha256'] + '\n')
    images = sorted(image_dir.glob('slide-*.png'), key=lambda p: int(p.stem.split('-')[-1]))
    # Filter stale pages if an earlier deck had more pages.
    images = [p for p in images if int(p.stem.split('-')[-1]) <= len(slides)]
    if len(images) != len(slides):
        raise RuntimeError('Rendered slide count does not match the narration.')
    playlist = args.workdir / 'slides.ffconcat'
    lines = ['ffconcat version 1.0']
    for record, path in zip(records, images):
        lines += [f'file {ffquote(path)}', f"duration {record['duration']:.9f}"]
    lines += [f'file {ffquote(images[-1])}']
    playlist.write_text('\n'.join(lines) + '\n')
    command = [ffmpeg, '-hide_banner', '-y', '-f', 'concat', '-safe', '0', '-i', playlist,
               '-i', narration_path, '-i', args.output.with_suffix('.srt'), '-i', metadata_path,
               '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0', '-map_metadata', '3', '-map_chapters', '3',
               '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1',
               '-r', str(args.fps), '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', str(args.threads),
               '-c:a', 'aac', '-b:a', '160k', '-af', 'loudnorm=I=-18:TP=-1.5:LRA=7',
               '-c:s', 'mov_text', '-disposition:s:0', '0', '-metadata:s:s:0', 'language=eng',
               '-metadata:s:a:0', 'title=Kokoro af_heart synthetic narration',
               '-t', f'{total:.9f}', '-movflags', '+faststart', args.output]
    print('Encoding 1080p H.264/AAC video with optional English captions...', flush=True)
    run(command)
    probe = subprocess.run([ffmpeg, '-hide_banner', '-i', str(args.output),
                            '-t', '0', '-f', 'null', '-'], capture_output=True, text=True)
    duration_match = re.search(r'Duration:\s+(\d+):(\d+):(\d+\.\d+)', probe.stderr)
    if probe.returncode or not duration_match:
        raise RuntimeError('Encoded video could not be decoded for validation.\n' + probe.stderr[-3000:])
    video_duration = sum(float(value) * weight for value, weight in zip(duration_match.groups(), [3600, 60, 1]))
    if abs(video_duration - total) > 0.25:
        raise RuntimeError(f'Encoded duration {video_duration} differs from the audio timeline {total}.')
    if not all(token in probe.stderr for token in ['Video: h264', 'Audio: aac', '1920x1080', 'mov_text']):
        raise RuntimeError('Encoded video does not have the requested video/audio/caption streams.')
    print('Decoding the complete video and audio streams for validation...', flush=True)
    run([ffmpeg, '-v', 'error', '-i', args.output, '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'])
    if sha256(args.guide) != guide_hash or sha256(args.pdf) != pdf_hash:
        raise RuntimeError('Guide or PDF changed during rendering. Output is not current; rerun using cached narration.')
    manifest.update({'video_created': True, 'video_sha256': sha256(args.output),
                     'video_bytes': args.output.stat().st_size, 'output': args.output.name,
                     'encoded_duration_seconds': video_duration,
                     'video_validation': 'complete H.264 1920x1080 and AAC streams decoded without errors; mov_text captions present; timeline within 0.25 seconds'})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Created {args.output.resolve()} ({args.output.stat().st_size/1024**2:.1f} MiB).', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
