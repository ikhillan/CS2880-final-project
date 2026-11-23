# Who gets a place?

Start with [the narrated presentation](presentation-narrated.mp4) or [the slides](presentation.pdf). The talk has **18 slides**, three empirical figures, and **17 minutes, 6 seconds of narration**. It follows one argument: changing other people's shortlists can change your access, and a higher match rate does not guarantee protection for each recipient.

- [Speaker notes](speaker-notes.pdf) and [editable narration](presenter-guide.txt) explain each comparison and the mathematical result in spoken language.
- [Technical appendix](technical-appendix.pdf) contains 25 slides: detailed jobs results, all rental panels and controls, uncertainty, data limitations, the retention formulation and proof, and institutional details.
- The [MP4](presentation-narrated.mp4) is included in the repository for team sharing. It presents the slides with a clearly disclosed **synthetic narrator**, optional English captions and slide chapters. It is not a recording or voice clone of any author. Speech-model files and intermediate rendering assets remain ignored.
- [SRT captions](presentation-narrated.srt), [WebVTT captions](presentation-narrated.vtt) and [narration manifest](narration-manifest.json) record the actual synthesized cue timings, voice/model hashes, source hashes and video checks.

The paper and experiments were not edited in this presentation revision.

## The argument

Slides 2–4 give a simple competition example, define the outcomes, and state the unchanged-shortlist finding. Slides 5–10 explain the settings and coordination rules, compare aggregate outcomes, and separate gains from lost matches. Slides 11–12 connect that access question to Chilean application reports and show why the comparator matters. Slide 13 states the retention problem and hardness result; slide 14 explains the reduction; slide 15 compares protection bounds; slide 16 tests prospective rules. Slide 17 addresses exclusion and proposed safeguards, and slide 18 closes with the next design question. Material qualifications accompany their results rather than occupying a separate limits slide.

The headline concerns 2,328 modeled renter appearances with one to five feasible apartments in the September–October enriched panel. Their match rate rises from **13.1% under the anchor to 27.5% under balancing**, a paired gain of 14.41 percentage points, although their own shortlists cannot change. This subgroup still has 15 exits. Across all 10,240 appearances in the panel, anchoring produces 449 entrants and 37 exits relative to GPT, for a net gain of 412. These are simulated allocation effects, not observed housing outcomes.

The main narration retains the comparisons that qualify this result. Balancing can edit several options while the anchor edits at most one. Balancing beats random on both mean outcomes, so the tradeoff is not universal. A model-informed score control improves both displayed outcomes further; the talk does not claim GPT superiority. Detailed jobs comparisons, including coordinated random's higher mean placement than coordinated GPT, remain in the appendix.

The main Chile comparison uses later enrollment as the coverage reference, although the classifiers predict initial offers. All changed-menu outcomes remain counterfactual under fixed reports. The theorem concerns a fixed-priority model on an explicit priority panel. The institutional tests separately handle family rules. The structural rule's +28 historical hits are a small prospective gain, not a large demonstrated social benefit.

## What changed after the review

The earlier talk repeatedly switched settings and deferred the prospective guarantee to the appendix. The revision leads with the unchanged-shortlist finding, keeps the rental explanation together, and uses it to motivate the Chilean retention problem. The main talk includes proof intuition, the modest prospective gain, and individual losses despite better totals. Secondary tables move to the appendix; counterresults that change the interpretation remain visible or spoken beside the relevant claim.

[The candid review](TALK_REVIEW.md) distinguishes surprising findings from established methods and unsupported claims. [The GitHub review](LATEST_GITHUB_REVIEW.md) verifies that Ishan's latest rental contribution was already included and documents property overlap across calendar windows. Those windows are not independent property-level validation; no cached experiment was silently changed.

## Figures and evidence

[make_figures.py](make_figures.py) reads released aggregate rows and generates five vector figures, matching PNGs and presentation tables. Three empirical figures appear in the timed talk. [figure-sources.json](figure-sources.json) records hashes for 12 aggregate sources. The rental policy scatter shows four observed policy means, not an estimated Pareto frontier. The paired intervals and all controls are in the appendix.

Primary sources inside the repository:

- [Reconstructed rental access evidence](../../findings/rental-anchor-access/README.md)
- [Original cached GPT rental results](../../findings/gpt-luna-rentals/README.md)
- [Verified Chile evidence](../../findings/chile/README.md)
- [Published jobs results and missing-transition bounds](../../findings/jobs-anchor-access/README.md)

## Rebuild the slides

With Python, Matplotlib and a TeX distribution containing Beamer, TikZ and Latin Modern:

```sh
python3 build.py
```

The build checks contiguous narration timings, source selections, slide/narration counts and LaTeX overflow, then creates the main PDF, appendix and notes. Visual inspection remains necessary. Build files, renders and model caches are ignored.

## Rebuild the narrated video

[render_video.py](render_video.py) uses local Kokoro speech synthesis and a working FFmpeg binary. It does not send the script to an external speech service or download dependencies automatically. The local generation used a separate `.build/tts/venv` with `kokoro-onnx`, `onnxruntime`, `soundfile`, `numpy` and `imageio-ffmpeg`.

Place `kokoro-v1.0.onnx` and `voices-v1.0.bin` under `.build/tts/`. Their exact hashes and source releases are in the narration manifest. The ordinary `af_heart` synthetic voice comes from the [Kokoro ONNX model release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1), using the [Kokoro-82M model](https://huggingface.co/hexgrad/Kokoro-82M). The model is Apache-2.0 licensed and the wrapper is MIT licensed.

```sh
.build/tts/venv/bin/python render_video.py --check
.build/tts/venv/bin/python render_video.py --speed 0.88 --timing guide
```

Audio chunks are cached by text, voice, model and synthesis settings. Each caption cue uses measured audio boundaries. The renderer does not truncate speech or automatically accelerate it. The final video uses H.264 at 1920×1080, AAC audio, optional English subtitles, and slide chapters. The manifest reports actual duration; the guide's nominal timestamps are not a substitute for that measurement.
