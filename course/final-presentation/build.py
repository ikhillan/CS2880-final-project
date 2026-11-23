"""Build the Beamer talk, technical appendix and timed narration PDF."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUILD = HERE / ".build"


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "—": "---", "–": "--", "−": "-", "’": "'"}
    return "".join(replacements.get(char, char) for char in text)


def make_notes():
    guide = (HERE / "presenter-guide.txt").read_text()
    parts = re.split(r"(?m)^SLIDE (\d+) — (.+)\n", guide)
    slide_count = len(re.findall(r"\\begin\{frame\}", (HERE / "main.tex").read_text()))
    if len(parts) != 1 + 3*slide_count:
        raise ValueError(f"Expected {slide_count} timed slide sections")
    preamble = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,geometry,xcolor,hyperref}
\geometry{letterpaper,margin=1in}
\definecolor{navy}{HTML}{243746}
\hypersetup{pdftitle={Timed narration: recommendations in matching markets},
pdfauthor={Ishan Khillan, Elaine Zhang, Todd Zhou}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{9pt}
\renewcommand{\familydefault}{\sfdefault}
\begin{document}
"""
    chunks = [preamble]
    seconds_total = 0
    words = 0
    for index in range(1, len(parts), 3):
        number, title, body = parts[index:index+3]
        timing, narration = body.strip().split("\n", 1)
        match = re.fullmatch(r"(\d{2}):(\d{2})–(\d{2}):(\d{2}) \| (\d+) seconds", timing)
        if not match:
            raise ValueError(f"Bad timing: {timing}")
        start_m, start_s, end_m, end_s, duration = map(int, match.groups())
        if start_m*60+start_s != seconds_total or end_m*60+end_s != seconds_total+duration:
            raise ValueError("Non-contiguous slide timing")
        seconds_total += duration
        words += len(narration.split())
        if int(number) != (index+2)//3:
            raise ValueError("Slide narration numbers are not sequential")
        if index > 1:
            chunks.append(r"\newpage")
        chunks.append(r"{\small Harvard CS 2880 / AI for Social Impact}\par")
        chunks.append(r"\section*{Slide " + number + ": " + tex_escape(title) + "}")
        chunks.append(r"{\color{navy}\textbf{" + tex_escape(timing) + r"}}\par\medskip")
        for para in re.split(r"\n\s*\n", narration.strip()):
            chunks.append(tex_escape(" ".join(para.split())) + "\n\\par")
    if not 960 <= seconds_total <= 1080:
        raise ValueError(f"Narration should be around 17 minutes: {seconds_total} seconds")
    chunks.append(r"\end{document}")
    (HERE / "speaker-notes.tex").write_text("\n".join(chunks)+"\n")
    print(f"Narration: {words} words, {slide_count} slides, {seconds_total//60}:{seconds_total%60:02d} planned duration.")


def compile_pdf(source, output):
    engine = shutil.which("pdflatex")
    if not engine:
        raise RuntimeError("Install a TeX distribution with pdflatex and Beamer")
    command = [engine, "-halt-on-error", "-interaction=nonstopmode", "-file-line-error",
               "-output-directory="+str(BUILD), source]
    for _ in range(2):
        result = subprocess.run(command, cwd=HERE, text=True, capture_output=True)
        if result.returncode:
            print(result.stdout[-8000:])
            raise RuntimeError(f"LaTeX failed for {source}")
    stem = Path(source).stem
    log = (BUILD / (stem+".log")).read_text(errors="replace")
    warnings = [line for line in log.splitlines() if "Overfull" in line]
    if warnings:
        raise RuntimeError(f"Overflow in {source}: " + "\n".join(warnings))
    shutil.copyfile(BUILD/(stem+".pdf"), HERE/output)
    print(HERE/output)


def main():
    BUILD.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(BUILD/"matplotlib"))
    subprocess.run([sys.executable, str(HERE/"make_figures.py")], check=True, env=env)
    make_notes()
    compile_pdf("main.tex", "presentation.pdf")
    compile_pdf("technical-appendix.tex", "technical-appendix.pdf")
    compile_pdf("speaker-notes.tex", "speaker-notes.pdf")


if __name__ == "__main__":
    main()
