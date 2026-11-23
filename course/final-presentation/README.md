# Who gets a place?

[Watch the presentation](presentation-narrated.mp4) · [Hear the three voices](presentation-voice-preview.mp3) · [Read the slides](presentation.pdf)

A group of renters in our simulation saw every apartment they could afford and qualify for under the model. Their own lists stayed fixed, but their match rate rose from **13.1% to 27.5%** when we changed other renters' shortlists. A person’s access can change even when the recommendations on their screen stay the same.

The **17-minute narrated presentation** for Harvard CS 2880 has 19 slides. It opens with the question on slide 1 and the result on slide 2. A two-seat example explains how an unchanged list can still cost someone a place. The main talk uses one empirical chart, two small comparison tables and seven timed reveals. The [28-slide technical appendix](technical-appendix.pdf) contains the estimates, additional baselines, proof details and field-study contrasts. The [speaker notes](speaker-notes.pdf), [editable script](presenter-guide.txt), [captions](presentation-narrated.vtt) and [media manifest](narration-manifest.json) accompany the video.

**Viewing guide.** The headline slide starts at 0:14. Both match rates are on screen by 0:30. Jump to 5:46 for who gained or lost an apartment, 6:44 for the jobs control, 7:59 for Chilean admissions, 9:36 for the algorithmic problem, 14:05 for the randomized field study or 15:50 for the ethical implications.

## What each experiment establishes

- **NYC rentals.** In the September–October enriched panel, 2,328 simulated renter appearances had one to five feasible listings and saw all of them under both rules. Balancing exposure across listings instead of anchoring one slot per shortlist produced 349 entrants and 15 exits in this group. Pooled matched counts rose from 304 to 638. The rates on screen average within-seed proportions. Across the three previously studied windows, subgroup gains ranged from 14.4 to 14.9 percentage points in enriched samples and 20.9 to 24.5 points in ordinary samples. The appendix reports each contrast and its pointwise paired interval. Properties recur, preferences and priorities are modeled, and the two policies have different edit budgets. [Rental audit](../../findings/rental-anchor-access/README.md)
- **NYC jobs.** In the replication, applying the anchor to GPT shortlists raised placement from 63.5% to 79.4%. The same anchoring rule applied to random shortlists reached 81.4%. The latter comparison is between published point means. No paired interval for it is released. The 612 net additional GPT placements and 920 changed menus bound gross exits between zero and 308 seeker appearances under the model. The individual transitions and post-anchor coverage are unknown. [Replication](../../findings/jobs-gpt-frozen/replication/anchor.csv) · [Derivation](../../findings/jobs-anchor-access/README.md)
- **Chilean admissions.** In a 2021 entry-grade replay, 237 of 309 net lost placements occurred among pupils whose shortlists stayed fixed. About 97% of the linked pupils were in that group to begin with. In the separate 2022–25 linked cohorts, pupils without a historical offer made up 7.4% of the sample and accounted for 76% of the net placement loss under the rank-informed versus rank-blind comparison. The comparison with first-five reverses the aggregate placement result. These counterfactuals hold families' reports fixed. [Chilean evidence](../../findings/chile/README.md)

An external randomized Chilean study compared a map with school information and highlights against a basic map. Qualifying second choices rose by 5.4 percentage points. Any assignment changed by an estimated +0.2 points, with an approximate 95% interval from −2.0 to +2.4. The trial and our replay evaluate different interventions. The next slide explains the difference. In a three-pupil example, the average effect on the treated pupil is +1/6 at quarter adoption, yet full rollout reduces placements from two to one. [Field study](https://doi.org/10.3386/w32670) · [Extracted contrasts](../../publication/two-studies/field-study/trials/explorer-contrasts.csv) · [Verifiable example](../../publication/two-studies/field-study/interference/README.md)

The algorithmic question is whether one shortlist policy can improve target coverage and retain every baseline recipient across a supplied panel of strict priorities. The talk proves NP-completeness for the decision problem using Independent Set. It also explains why preserving a seat can allow more coverage gains than preserving the exact assignment. On Chile's 2022–24 design panels, exact-contract protection permits 314 additional historical hits. Allowing switches admits a policy with 642 hits and gives an upper bound of 4,620. A separate prospective structural rule gains 28 hits and guarantees retention on permitted ordinary lottery orders in the reconstructed mechanism. It adds no entrants in the reported paired tests. [Theorem](../../publication/two-studies/retention-theory.tex) · [Results](../../findings/chile/README.md)

[Figure provenance](figure-sources.json) records the 16 released aggregate and analysis inputs. This presentation uses existing experiments. [Assessment of the claims](TALK_REVIEW.md)

## Narration

The video uses three disclosed synthetic U.S. English voices. Jane presents the opening, theory and conclusion. Michael presents rentals and jobs. George presents the Chilean results and social implications. Each complete passage plays at **1.45× its synthesized tempo**, preserving pitch. The rate is relative to the approved three-voice audition. Pauses between thoughts and at slide changes are timed separately. The manifest records the measured run time, source and paced durations, voice assignments, clip hashes, caption alignment and video checks.

Kyutai's presets use recordings from the CSTR VCTK Corpus 0.92 by Junichi Yamagishi, Christophe Veaux and Kirsten MacDonald (2019), released under CC BY 4.0. The corpus lists p339 as female, American, Pennsylvania, p360 as male, American, New Jersey, and p315 as male, American, New England. Pocket TTS uses these source voices to synthesize the authors' written narration locally. The public model weights and voice references are CC BY 4.0. [Voice credits](https://huggingface.co/kyutai/tts-voices#vctk) · [Corpus](https://doi.org/10.7488/ds/2645) · [Speaker metadata](https://storage.googleapis.com/tfds-data/dataset_info/vctk/mic1/1.0.1/metadata.json) · [Public checkpoint](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) · [License](https://creativecommons.org/licenses/by/4.0/)

## Rebuild

Use a Python environment with Matplotlib and a TeX distribution with Beamer and Latin Modern. From this directory:

```sh
python3 build.py
```

The video workflow uses Pocket TTS 4.0.1 for synthesis, FFmpeg for pitch-preserving tempo and encoding, and local Whisper tiny.en for caption alignment. The voice helper fetches the pinned public weights on first use. Model files, audio intermediates and speech-recognition diagnostics stay under `.build/`. The renderer needs Python with NumPy, soundfile and imageio-ffmpeg, plus Poppler on PATH. The alignment step needs openai-whisper, SciPy and PyTorch. Frame verification needs Pillow and a working FFmpeg on PATH, in imageio-ffmpeg, or supplied with --ffmpeg. The recorded versions and hashes are in the manifest.

```sh
# The first command reproduces the committed passage plan from the guide.
.build/pocket/venv/bin/python synthesize_us_voices.py --prepare-passages --roles narration-speakers.json
.build/pocket/venv/bin/python synthesize_us_voices.py --skip-previews
.build/tts/venv/bin/python pace_recordings.py \
  --recordings .build/pocket/recordings/recordings.json \
  --output-dir .build/us-paced --tempo 1.45

# With the published tiny.en checkpoint at the path below:
python3.12 align_captions.py --recordings .build/us-paced/recordings.json \
  --model .build/tts/tiny.en.pt --output caption-timings.json
.build/tts/venv/bin/python render_video.py --recordings .build/us-paced/recordings.json \
  --caption-timings caption-timings.json --timing guide \
  --lead-pause 0.45 --tail-pause 1.0 --sentence-pause 0.55 \
  --build-manifest video-builds.json --build-pdf video-builds.pdf \
  --workdir .build/us-render --output presentation-narrated.mp4

# Independently match three decoded frames per visual build to the source pages.
python3 verify_presentation.py
```

Pitch-preserving processing uses the same factor for every speaker. The alignment step fits the exact written script into two-line captions and estimates phrase onsets within each measured passage. Build changes use the measured start of their actual audio passage. The final renderer checks script identity, cue order, input hashes, stream formats and run time, then decodes the entire video and audio.
