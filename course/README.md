# CS 2880 presentation and course archive

The current presentation is **Who gets a place?**, an 18-slide talk of approximately 17 minutes for Harvard's AI for Social Impact course. Its main finding is that a group's match rate rises from 13.1% to 27.5% when other people's shortlists change, even though the group's own shortlists remain identical. These are modeled rental outcomes; the talk connects them to recipient protection in Chilean admissions.

- [Narrated presentation (MP4)](final-presentation/presentation-narrated.mp4), included for team sharing, with a disclosed synthetic voice, optional captions and slide chapters.
- [Main slides (PDF)](final-presentation/presentation.pdf)
- [Speaker notes (PDF)](final-presentation/speaker-notes.pdf) and [editable narration](final-presentation/presenter-guide.txt)
- [Technical appendix (PDF)](final-presentation/technical-appendix.pdf), with detailed comparisons, uncertainty, model controls and proofs.
- [Presentation sources, evidence and rebuild instructions](final-presentation/README.md)

The [current working paper](../publication/two-studies/who-gets-a-place.pdf) and [manuscript sources](../publication/two-studies/README.md) are separate artifacts. The presentation revision did not change the paper or experiments. [Current findings](../findings/README.md) document the evidence used in the talk.

## Earlier course materials

These dated documents show how the original brokerage question developed into the shortlist study. Their amounts, experiments and future-tense plans describe their respective stages; they are not the final presentation.

| Course artifact | Contents |
| --- | --- |
| [Original proof sketch](proof-sketch/proof-sketch.tex) | October 25 working note. Gives the initial model and a hardness reduction for selecting disclosure actions. The standard deferred acceptance algorithm itself runs in polynomial time. |
| [Faculty feedback presentation](faculty-feedback/faculty-feedback.pdf) | October 29 Beamer discussion of the original matching model. Editable source and speaker notes are in the same folder. |
| [Archived October 30 presentation](course-presentation/full-paper-slides.pdf) | Earlier synthetic study, with the corrected results table on page 22. Timeouts mean the algorithm failed to certify the minimum within ten seconds. The supplement contains 17 additional slides. |
| [NYC experiment protocol](rental-experiment-protocol.md) | November 7 plan including the November 12 development amendment. The completed studies and their individual protocols are documented with the current paper. |

The archived presentations compile with `latexmk -pdf main.tex` in their directories. To compile the proof note, run `latexmk -pdf proof-sketch.tex` in its directory. Their presentation data files summarize the earlier synthetic experiments. The current presentation instead uses `python3 build.py` in `final-presentation/`; its README gives the full workflow.
