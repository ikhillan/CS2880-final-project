# CS 2880 presentation and course archive

**Who gets a place?** is a 21-slide, 17-minute, 40-second presentation for Harvard's AI for Social Impact course. In the NYC rental simulation, other people's recommendations changed a fixed-shortlist subgroup's placement rate from 13.1% to 27.5%. The talk examines how this can happen, tests a coordinated jobs baseline, and uses Chilean admissions to ask which recipients an algorithm can protect.

- [Narrated video](final-presentation/presentation-narrated.mp4) and [short American voice preview](final-presentation/presentation-voice-preview.mp3)
- [Main slides](final-presentation/presentation.pdf) and [technical appendix](final-presentation/technical-appendix.pdf)
- [Speaker notes](final-presentation/speaker-notes.pdf) and [editable script](final-presentation/presenter-guide.txt)
- [Evidence, cast credits and video rebuild instructions](final-presentation/README.md)
- [Working paper](../publication/two-studies/who-gets-a-place.pdf) and [research findings](../findings/README.md)

## Earlier course materials

These dated documents show how the original brokerage question developed into the shortlist study. Their amounts, experiments and future-tense plans describe their respective stages; they are not the final presentation.

| Course artifact | Contents |
| --- | --- |
| [Original proof sketch](proof-sketch/proof-sketch.tex) | October 25 working note. Gives the initial model and a hardness reduction for selecting disclosure actions. The standard deferred acceptance algorithm itself runs in polynomial time. |
| [Faculty feedback presentation](faculty-feedback/faculty-feedback.pdf) | October 29 Beamer discussion of the original matching model. Editable source and speaker notes are in the same folder. |
| [Archived October 30 presentation](course-presentation/full-paper-slides.pdf) | Earlier synthetic study, with the corrected results table on page 22. Timeouts mean the algorithm failed to certify the minimum within ten seconds. The supplement contains 17 additional slides. |
| [NYC experiment protocol](rental-experiment-protocol.md) | November 7 plan including the November 12 development amendment. The completed studies and their individual protocols are documented with the current paper. |

The archived presentations compile with `latexmk -pdf main.tex` in their directories. To compile the proof note, run `latexmk -pdf proof-sketch.tex` in its directory. Their presentation data files summarize the earlier synthetic experiments. The current presentation instead uses `python3 build.py` in `final-presentation/`; its README gives the full workflow.
