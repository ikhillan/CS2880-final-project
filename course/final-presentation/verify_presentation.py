#!/usr/bin/env python3
"""Independent local checks for the frozen narration/video artifact."""
from pathlib import Path
import argparse
import io
import json
import subprocess
from PIL import Image, ImageChops, ImageOps, ImageStat

from render_video import find_ffmpeg, sha256

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'narration-manifest.json'
VIDEO = ROOT / 'presentation-narrated.mp4'
QA = ROOT / '.build/us-frame-qa'
QA.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ffmpeg', help='Path to a working FFmpeg executable')
args = parser.parse_args()

m = json.loads(MANIFEST.read_text())
assert m['video_created'] is True
assert m['video_sha256'] == sha256(VIDEO)
assert m['guide_sha256'] == sha256(ROOT / 'presenter-guide.txt')
assert m['pdf_sha256'] == sha256(ROOT / 'presentation.pdf')
if m.get('build_pdf_sha256'):
    assert m['build_pdf_sha256'] == sha256(ROOT / 'video-builds.pdf')
    assert m['build_manifest_sha256'] == sha256(ROOT / 'video-builds.json')
assert m['renderer_sha256'] == sha256(ROOT / 'render_video.py')
assert abs(m['encoded_duration_seconds'] - m['actual_total_seconds']) < .05
assert m['actual_total_seconds'] == m['planned_total_seconds']
plan = json.loads((ROOT / 'narration-speakers.json').read_text())
assert len(m['slides']) == len(plan)
assert all(r['overrun_seconds'] == 0 for r in m['slides'])
assert m['automatic_speed_adjustment'] is False
assert m['requested_playback_tempo'] == 1.45
assert all(r['speech_speed'] == 1.45 for r in m['slides'])
assert m['caption_timings_sha256'] == sha256(ROOT / 'caption-timings.json')
assert m['passage_source_sha256'] == sha256(ROOT / 'narration-passages.json')
assert m['speech_truncated'] is False
assert m['full_text_and_caption_boundaries_verified'] is True
assert all(r['speaker'] == plan[str(r['slide'])] for r in m['slides'])
assert set(m['voices']) == {'jane','michael','george'}

ffmpeg, _ = find_ffmpeg(args.ffmpeg)
sources = sorted((ROOT / '.build/us-render/slides').glob('slide-*.png'),
                 key=lambda p: int(p.stem.split('-')[-1]))
segments = m.get('visual_segments', m['slides'])
assert len(sources) == len(segments)
thumbnails = [ImageOps.pad(Image.open(p).convert('RGB'), (1920, 1080),
                          color='white').resize((320, 180)) for p in sources]
checks = []
for r in segments:
    page = r.get('page', r['slide'])
    for position, stamp in [
            ('start', r['start'] + 0.25),
            ('middle', (r['start'] + r['end']) / 2),
            ('end', r['end'] - 0.25)]:
        data = subprocess.check_output([
            ffmpeg, '-v', 'error', '-ss', str(stamp), '-i', str(VIDEO),
            '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'
        ])
        frame = Image.open(io.BytesIO(data)).convert('RGB')
        assert frame.size == (1920, 1080)
        thumbnail = frame.resize((320, 180))
        errors = [sum(ImageStat.Stat(ImageChops.difference(thumbnail, t)).mean) / 3
                  for t in thumbnails]
        best = min(range(len(errors)), key=errors.__getitem__) + 1
        assert best == page, (page, position, best, errors)
        checks.append({'slide': r['slide'], 'page': page, 'position': position,
                       'time_seconds': stamp, 'matched_source_page': best,
                       'mean_absolute_rgb_error_320x180': errors[best-1]})
        if position == 'middle':
            frame.save(QA / f"page-{page:02}.png")
    print(f"Page {page:02}, slide {r['slide']:02}: source matches at start, middle and end", flush=True)

summary = {'video_sha256': m['video_sha256'], 'frame_count': len(checks),
           'all_visual_build_sources_match': True,
           'logical_slides': len(m['slides']), 'visual_builds': len(segments),
           'checked_positions': '0.25 seconds after start, midpoint, 0.25 seconds before end',
           'samples': checks}
(QA / 'frame-validation.json').write_text(json.dumps(summary, indent=2) + '\n')
m['post_render_frame_validation'] = summary
MANIFEST.write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps({'bytes': VIDEO.stat().st_size, 'sha256': sha256(VIDEO),
                  'duration_seconds': m['encoded_duration_seconds'],
                  'caption_cues': m['caption_count'],
                  'frames_checked': len(checks),
                  'maximum_frame_error': max(c['mean_absolute_rgb_error_320x180']
                                             for c in checks)}, indent=2))
