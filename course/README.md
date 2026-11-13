# Earlier CS 2880 course materials

These dated course documents show how a question about brokerage and hidden options became a study of shortlist design. For the final results, updated assumptions, and full replication materials, see the [current paper](../publication/ACM_EC_RECOVERY_AND_PLACEMENT.pdf). The amounts, experiments, and future-tense plans in the slides and protocol describe their respective stages.

| Course artifact | Contents |
| --- | --- |
| [Original proof sketch](proof-sketch/proof-sketch.tex) | October 25 working note. Gives the initial model and a hardness reduction for selecting disclosure actions. The standard deferred acceptance algorithm itself runs in polynomial time. |
| [Faculty feedback presentation](faculty-feedback/faculty-feedback.pdf) | October 29 Beamer discussion of the original matching model. Editable source and speaker notes are in the same folder. |
| [Full course presentation](course-presentation/full-paper-slides.pdf) | October 30 revision of the earlier synthetic study, with the corrected results table on page 22. Timeouts mean the algorithm failed to certify the minimum within ten seconds. The supplement contains 17 additional slides. |
| [NYC experiment protocol](rental-experiment-protocol.md) | November 7 plan including the November 12 development amendment. The completed studies and their individual protocols are documented with the current paper. |

Each presentation compiles with `latexmk -pdf main.tex` in its directory. To compile the proof note, run `latexmk -pdf proof-sketch.tex` in its directory. Presentation data files are synthetic experiment summaries.
