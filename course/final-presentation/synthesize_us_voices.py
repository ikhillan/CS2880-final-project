#!/usr/bin/env python3
"""Record complete narration passages with pinned, public Pocket TTS presets.

Run with .build/pocket/venv/bin/python synthesize_us_voices.py. Model files,
previews and recordings stay in .build/pocket. No uploaded or cloned voice is
accepted. The resulting recordings.json is an input to render_video.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import time
import wave

HERE = Path(__file__).resolve().parent
REPO = 'kyutai/pocket-tts-without-voice-cloning'
REVISION = 'a7056f505a4a8ca72d8b5fb1440071d6f1a35739'
TOKENIZER_REVISION = '00eac05ed3d16bdc3f6b5d598874019c34a89214'
VOICES = {
    'jane': {'vctk_id': 'p339', 'gender': 'female', 'accent': 'American', 'region': 'Pennsylvania'},
    'mary': {'vctk_id': 'p333', 'gender': 'female', 'accent': 'American', 'region': 'Indiana'},
    'michael': {'vctk_id': 'p360', 'gender': 'male', 'accent': 'American', 'region': 'New Jersey'},
    'george': {'vctk_id': 'p315', 'gender': 'male', 'accent': 'American', 'region': 'New England'},
}
AUDITION = (
    "Here's the result that surprised us. Their personal shortlists stayed exactly the same, "
    "but their match rate rose from thirteen point one to twenty-seven point five percent. "
    "Only the recommendations that everyone else received had changed."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def cardinal(n: int) -> str:
    units = ('zero one two three four five six seven eight nine ten eleven twelve '
             'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
    tens = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
    if n < 20:
        return units[n]
    if n < 100:
        return tens[n // 10] + ('-' + units[n % 10] if n % 10 else '')
    for scale, name in [(1_000_000, 'million'), (1000, 'thousand'), (100, 'hundred')]:
        if n >= scale:
            return cardinal(n // scale) + ' ' + name + (' ' + cardinal(n % scale) if n % scale else '')
    raise ValueError(n)


def pronunciation_text(text: str) -> str:
    """Spell numeric quantities without changing the exact caption/lookup text."""
    def number(match):
        value = match.group().replace(',', '')
        whole, dot, fraction = value.partition('.')
        return cardinal(int(whole)) + (' point ' + ' '.join(cardinal(int(c)) for c in fraction) if dot else '')
    return re.sub(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)', number, text)


def read_passages(path: Path, expected_clips: int | None = None):
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty list of reviewed narration passages.')
    if expected_clips is not None and len(rows) != expected_clips:
        raise ValueError(f'Expected exactly {expected_clips} narration passages, found {len(rows)}.')
    if len({(r['slide'], r['index']) for r in rows}) != len(rows):
        raise ValueError('Duplicate passage index.')
    if len({(r['slide'], r['text']) for r in rows}) != len(rows):
        raise ValueError('Duplicate slide/text lookup key.')
    slide_count = max(r['slide'] for r in rows)
    if {r['slide'] for r in rows} != set(range(1, slide_count + 1)):
        raise ValueError('Passages must cover contiguous slides starting at 1.')
    if [(r['slide'], r['index']) for r in rows] != sorted((r['slide'], r['index']) for r in rows):
        raise ValueError('Passages must be in slide and passage order.')
    for slide in range(1, slide_count + 1):
        indices = [r['index'] for r in rows if r['slide'] == slide]
        if indices != list(range(len(indices))):
            raise ValueError(f'Noncontiguous passage indices on slide {slide}.')
        if len({r['role'] for r in rows if r['slide'] == slide}) != 1:
            raise ValueError(f'A slide cannot change speaker mid-slide: {slide}.')
    for row in rows:
        if row['role'] not in ('female', 'male1', 'male2') or not row['text'].strip() or not row['spoken_text'].strip():
            raise ValueError(f'Invalid passage: {row}')
    return rows, hashlib.sha256(raw).hexdigest()


def write_wav(path: Path, samples, rate=24000):
    import numpy as np
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Empty or nonfinite audio.')
    peak = float(np.max(np.abs(samples)))
    if peak < 0.001:
        raise ValueError('Silent synthesis.')
    # Only attenuate an over-range waveform; never clip samples or time-stretch.
    gain = min(1.0, 0.98 / peak)
    pcm = (samples * gain * 32767).round().astype('<i2')
    tmp = path.with_suffix('.wav.tmp')
    with wave.open(str(tmp), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    tmp.replace(path)
    return {'duration_seconds': len(samples) / rate, 'samples': len(samples),
            'sample_rate': rate, 'peak_before_attenuation': peak, 'gain': gain,
            'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passages', type=Path, default=HERE / 'narration-passages.json')
    parser.add_argument('--prepare-passages', action='store_true', help='Extract deterministic renderer-compatible passages, then exit without model loading.')
    parser.add_argument('--guide', type=Path, default=HERE / 'presenter-guide.txt')
    parser.add_argument('--builds', type=Path, default=HERE / 'video-builds.json')
    parser.add_argument('--roles', type=Path, help='For --prepare-passages: explicit slide-to-role JSON, optionally under a slides key.')
    parser.add_argument('--workdir', type=Path, default=HERE / '.build/pocket')
    parser.add_argument('--female', choices=['jane', 'mary'], default='jane')
    parser.add_argument('--expected-clips', type=int, help='Optional exact completeness check for this script revision.')
    parser.add_argument('--audition-only', action='store_true')
    parser.add_argument('--skip-previews', action='store_true')
    parser.add_argument('--seed', type=int, default=2880)
    parser.add_argument('--regenerate', action='append', default=[], metavar='SLIDE:INDEX')
    args = parser.parse_args()
    if args.prepare_passages:
        if not args.roles:
            parser.error('--prepare-passages requires --roles')
        role_data = json.loads(args.roles.read_text())
        slide_roles = role_data.get('slides', role_data)
        renderer = runpy.run_path(str(HERE / 'render_video.py'))
        slides = renderer['read_guide'](args.guide)
        builds = json.loads(args.builds.read_text())['builds']
        if set(slide_roles) != {str(s.number) for s in slides}:
            raise ValueError('The role plan must assign every logical slide exactly once.')
        if {b['slide'] for b in builds} != {s.number for s in slides}:
            raise ValueError('Builds and narration disagree about logical slides.')
        passages = []
        for slide in slides:
            anchors = [b['anchor'] for b in builds if b['slide'] == slide.number and b['anchor'] is not None]
            for index, text in enumerate(renderer['narration_passages'](slide.text, anchors)):
                passages.append({'slide': slide.number, 'index': index, 'text': text,
                                 'spoken_text': renderer['spoken_text'](text), 'role': slide_roles[str(slide.number)]})
        args.passages.write_text(json.dumps(passages, indent=2, ensure_ascii=False) + '\n')
        read_passages(args.passages, args.expected_clips)
        print(f'Prepared {len(passages)} passages for {len(slides)} slides: {args.passages.name}')
        return
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(args.workdir / 'hf-cache')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for name in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN']:
        os.environ.pop(name, None)

    import numpy as np
    import pocket_tts
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

    version = importlib.metadata.version('pocket-tts')
    if version != '4.0.1':
        raise ValueError(f'Expected reviewed pocket-tts==4.0.1, found {version}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config_path = Path(pocket_tts.__file__).parent / 'config/english.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    expected = f'hf://{REPO}/languages/english/model.safetensors@{REVISION}'
    if cfg['weights_path_without_voice_cloning'] != expected:
        raise ValueError('Package English checkpoint pin changed.')
    for voice, info in VOICES.items():
        expected_origin = f'hf://kyutai/tts-voices/vctk/{info["vctk_id"]}_023_enhanced.wav'
        if _ORIGINS_OF_PREDEFINED_VOICES[voice] != expected_origin:
            raise ValueError(f'Unexpected upstream preset mapping for {voice}.')

    def asset(filename, revision=REVISION):
        return Path(hf_hub_download(REPO, filename, revision=revision, token=False))

    print('Downloading only pinned public no-cloning weights and preset states.', flush=True)
    model_card = asset('README.md')
    card_metadata = yaml.safe_load(model_card.read_text().split('---', 2)[1])
    if card_metadata.get('license') != 'cc-by-4.0':
        raise ValueError('Public checkpoint license no longer matches the reviewed CC-BY-4.0 card.')
    weights = asset('languages/english/model.safetensors')
    tokenizer = asset('languages/english/tokenizer.json', TOKENIZER_REVISION)
    cfg['weights_path'] = str(weights)
    cfg['weights_path_without_voice_cloning'] = None
    cfg['flow_lm']['lookup_table']['tokenizer_path'] = str(tokenizer)
    local_config = args.workdir / 'english-public-only.yaml'
    local_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    voice_paths = {voice: asset(f'languages/english/embeddings/{voice}.safetensors') for voice in VOICES}
    print('Loading public model on CPU; one intra-op and one inter-op thread.', flush=True)
    model = TTSModel.load_model(config=local_config, temp=0.3)
    model.has_voice_cloning = False
    if model.sample_rate != 24000:
        raise ValueError(f'Unexpected sample rate {model.sample_rate}')
    states = {voice: model.get_state_for_audio_prompt(path) for voice, path in voice_paths.items()}
    print('All four public preset states loaded.', flush=True)
    metadata = {
        'engine': 'pocket-tts', 'engine_version': version, 'torch_version': torch.__version__,
        'source_url': 'https://github.com/kyutai-labs/pocket-tts',
        'model_url': 'https://huggingface.co/' + REPO, 'model_revision': REVISION,
        'model_card_sha256': sha256(model_card),
        'model_sha256': sha256(weights), 'tokenizer_revision': TOKENIZER_REVISION,
        'tokenizer_sha256': sha256(tokenizer), 'package_config_sha256': sha256(config_path),
        'effective_config_sha256': sha256(local_config), 'generator_sha256': sha256(Path(__file__)),
        'model_license': 'CC-BY-4.0', 'wrapper_license': 'MIT', 'voice_license': 'CC-BY-4.0',
        'voice_source_url': 'https://huggingface.co/kyutai/tts-voices#vctk',
        'vctk_source_url': 'https://doi.org/10.7488/ds/2645',
        'vctk_speaker_metadata_url': 'https://storage.googleapis.com/tfds-data/dataset_info/vctk/mic1/1.0.1/metadata.json',
        'vctk_speaker_metadata_sha256': '98ccead73bd44a5f920dee49a1630f95f5d7a1db45537f8e4eba7d0c7188922c',
        'voice_attribution': 'Kyutai preset states from CSTR VCTK Corpus 0.92, Junichi Yamagishi, Christophe Veaux and Kirsten MacDonald (2019), CC BY 4.0. Enhanced source recordings provided by Kyutai.',
        'voices': {voice: {**info, 'state_sha256': sha256(voice_paths[voice]),
                          'preset_source': _ORIGINS_OF_PREDEFINED_VOICES[voice]} for voice, info in VOICES.items()},
        'temperature': 0.3, 'device': 'cpu', 'intraop_threads': 1, 'interop_threads': 1,
        'voice_cloning': False, 'authentication': 'Public downloads with token=False; no implicit credentials.',
        'generation_unit': 'Reviewed full phrase or paragraph; no sentence-level stitching inside a passage.',
        'postprocessing': 'PCM16 conversion and attenuation only if needed; no truncation or time-stretching.',
        'identity': 'Public synthetic narration presets; speakers do not impersonate the authors.',
    }
    roles = {'female': args.female, 'male1': 'michael', 'male2': 'george'}

    def synthesize(voice, text, seed, destination):
        torch.manual_seed(seed)
        start = time.monotonic()
        samples = model.generate_audio(states[voice], text, copy_state=True).detach().cpu().numpy()
        row = write_wav(destination, samples)
        row.update({'generation_seconds': time.monotonic() - start, 'seed': seed})
        return row, samples

    if not args.skip_previews:
        preview_dir = args.workdir / 'previews'
        preview_dir.mkdir(exist_ok=True)
        previews, parts = [], []
        for voice in [args.female, 'michael', 'george']:
            path = preview_dir / f'pocket-{voice}-us.wav'
            row, samples = synthesize(voice, AUDITION, args.seed, path)
            previews.append({'voice': voice, 'text': AUDITION, 'file': path.name, **row})
            parts.extend([samples.reshape(-1), np.zeros(24000, dtype=np.float32)])
            print('PREVIEW ' + json.dumps(previews[-1]), flush=True)
        combined = preview_dir / 'pocket-american-cast.wav'
        write_wav(combined, np.concatenate(parts[:-1]))
        atomic_json(preview_dir / 'previews.json', {'engine_metadata': metadata, 'clips': previews, 'combined': combined.name})
        print('COMBINED_PREVIEW_READY ' + str(combined), flush=True)
        other = 'mary' if args.female == 'jane' else 'jane'
        row, _ = synthesize(other, AUDITION, args.seed, preview_dir / f'pocket-{other}-us.wav')
        previews.append({'voice': other, 'text': AUDITION, 'file': f'pocket-{other}-us.wav', **row})
        atomic_json(preview_dir / 'previews.json', {'engine_metadata': metadata, 'clips': previews, 'combined': combined.name})
    if args.audition_only:
        return

    # Reopen only after auditions so approved textual edits reach the full run.
    rows, input_hash = read_passages(args.passages, args.expected_clips)
    plan = {'version': 1, 'speakers': {role: {'voice': voice, 'lang': 'en-us', 'speed': 1.0}
                                    for role, voice in roles.items()},
            'slides': {str(r['slide']): r['role'] for r in rows}}
    out = args.workdir / 'recordings'
    out.mkdir(exist_ok=True)
    manifest_path = out / 'recordings.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_rows = {(r['slide'], r['index']): r for r in previous.get('clips', [])}
    manifest = {'version': 1, 'status': 'in_progress', 'expected_clips': len(rows),
                'input_sha256': input_hash, 'input_file': args.passages.name,
                'engine_metadata': metadata, 'speaker_plan': plan, 'clips': []}
    recipe = {key: metadata[key] for key in ['engine_version', 'torch_version', 'model_sha256',
               'tokenizer_sha256', 'generator_sha256', 'temperature']}
    start = time.monotonic()
    for row in rows:
        if sha256(args.passages) != input_hash:
            raise RuntimeError('Passage input changed during synthesis; rerun to regenerate changed clips.')
        voice = roles[row['role']]
        synthesis_text = pronunciation_text(row['spoken_text'])
        seed = args.seed + row['slide'] * 100 + row['index']
        key = hashlib.sha256(json.dumps({**recipe, 'voice': voice, 'text': synthesis_text,
               'state_sha256': metadata['voices'][voice]['state_sha256'], 'seed': seed}, sort_keys=True).encode()).hexdigest()
        name = f'slide-{row["slide"]:02d}-{row["index"]:02d}-{voice}-{key[:12]}.wav'
        path = out / name
        old = previous_rows.get((row['slide'], row['index']))
        refresh = f'{row["slide"]}:{row["index"]}' in args.regenerate
        if old and old.get('recipe_sha256') == key and path.exists() and sha256(path) == old['sha256'] and not refresh:
            clip = {**old, **row}
        else:
            info, _ = synthesize(voice, synthesis_text, seed, path)
            clip = {**row, 'synthesis_text': synthesis_text, 'voice': voice, 'file': name,
                    'recipe_sha256': key, **info}
        manifest['clips'].append(clip)
        atomic_json(manifest_path, manifest)
        print('CLIP ' + json.dumps({'done': len(manifest['clips']), 'of': len(rows),
              'slide': row['slide'], 'index': row['index'], 'voice': voice,
              'duration_seconds': clip['duration_seconds'], 'file': name}), flush=True)
    if sha256(args.passages) != input_hash:
        raise RuntimeError('Input changed before finalization.')
    manifest.update({'status': 'complete', 'total_audio_seconds': sum(c['duration_seconds'] for c in manifest['clips']),
                     'run_seconds': time.monotonic() - start})
    atomic_json(manifest_path, manifest)
    print('RECORDINGS_COMPLETE ' + str(manifest_path), flush=True)


if __name__ == '__main__':
    main()
