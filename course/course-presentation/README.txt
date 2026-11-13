Stable matching with hidden options
Harvard COMPSCI 2880
Ishan Khillan, Elaine Zhang, Todd Zhou
October 29, 2026

Presentation
Open full-paper-slides.pdf. The main talk has 27 slides, followed by 17
appendix slides with proofs, additional results, proposed work, and references.
Allow roughly 30-35 minutes for the full talk. presenter-guide.txt includes
slide-by-slide explanations, a shorter route, and answers to likely questions.

Editing
main.tex includes the files in sections/. The matching diagrams use editable
TikZ. make_figures.py generates the scientific figure and four tables
from data/results.json.

Build with TeX Live:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

To regenerate figures and tables, install requirements-figures.txt and run:
  python3 make_figures.py
Then compile again. The compiler writes main.pdf.

Scope
The completed work is theoretical and computational. All displayed experimental
results are synthetic. The proposed listing study has not been run.

The deck accompanies the manuscript Disclosure Can Destabilize, stored in
outputs/hidden-blocking-pairs/. It includes the later singleton-firm and
common-priority observations. The manuscript and recorded experiments are
unchanged. Publication novelty still requires independent review.

Data
data/provenance.json records hashes of the source summaries. The complete
solver and raw instances remain in the earlier research package.

Generative AI assisted the analysis and drafting.
