#!/usr/bin/env python3
"""Regenerate the sibling handout's anchor table from Assignment A's .aux files.

Run after building Assignment A:

    cd docs/handout && latexmk -xelatex assignment-a.tex && python3 sync-xrefs.py

Reading the .aux is deliberate. The first design had \\anchor write the table
directly via \\immediate\\write to a \\newwrite stream; with biblatex, hyperref,
tcolorbox and six \\include'd files all competing for output streams, that
silently dropped later entries and duplicated earlier ones. The .aux already
carries the section number and the section title, and it is produced by LaTeX's
own label machinery, so it is both complete and correct.

Writes into the sibling repository. If that repository is not checked out next
to this one, the script says so and exits without touching anything -- neither
handout is allowed to require the other in order to build.
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A_AUX = os.path.join(HERE, "sections", "*.aux")
SIBLING = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "LLM-system-project", "docs", "handout")
)
OUT = os.path.join(SIBLING, "xref-a.tex")

HEADER = """% Stable anchors exported by Assignment A (llm-serving/docs/handout/).
%
% This is a COPY, not an include. The two handouts live in separate git
% repositories, so neither may depend on the other being checked out; xr-hyper
% would need the sibling's .aux at build time and would break a standalone clone.
%
% GENERATED -- do not hand-edit. Rebuild Assignment A, then run
%   llm-serving/docs/handout/sync-xrefs.py
% An \\\\extref to a key missing from this list warns at build time and typesets
% [??: key], so drift shows up in the log rather than silently.
"""

# \\newlabel{a:key}{{2.4}{11}{title}{subsection.2.4}{}}
PAT = re.compile(
    r"\\newlabel\{(a:[^}@]+)\}\{\{([^}]*)\}\{[^}]*\}\{(.*?)\}\{[^}]*\}\{\}\}"
)


def main():
    aux_files = sorted(glob.glob(A_AUX))
    if not aux_files:
        sys.exit("no .aux files -- build assignment-a.tex first")

    anchors = {}
    for path in aux_files:
        with open(path, encoding="utf8") as fh:
            for m in PAT.finditer(fh.read()):
                anchors[m.group(1)] = (m.group(2), m.group(3))

    if not anchors:
        sys.exit("no a: anchors found in the .aux files")

    if not os.path.isdir(SIBLING):
        print(f"sibling handout not checked out at {SIBLING}; nothing written.")
        print("Anchors that would have been exported:")
        for key in sorted(anchors):
            print(f"  {key} -> {anchors[key][0]}")
        return

    with open(OUT, "w", encoding="utf8") as fh:
        fh.write(HEADER)
        for key in sorted(anchors):
            num, title = anchors[key]
            fh.write("\\DeclareExternalAnchor{%s}{%s}{%s}\n" % (key, num, title))

    print(f"wrote {len(anchors)} anchors to {OUT}")
    for key in sorted(anchors):
        print(f"  {key:32s} -> A section {anchors[key][0]}")


main()
