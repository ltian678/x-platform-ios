r"""Materialize the public branch of the manuscript's \iffacebook switches.

Run: python -m xplat.public.tex INPUT.tex OUTPUT.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"(?<!\\newif)\\iffacebook|\\else|\\fi")


def public_branch(source: str) -> str:
    active = True
    stack: list = []
    output: list = []
    position = 0

    for match in TOKEN.finditer(source):
        if active:
            output.append(source[position : match.start()])

        token = match.group(0)
        if token == r"\iffacebook":
            stack.append((active, False))
            active = False
        elif token == r"\else":
            if not stack:
                raise ValueError(r"Unmatched \else")
            parent_active, condition = stack[-1]
            stack[-1] = (parent_active, condition)
            active = parent_active and not condition
        else:
            if not stack:
                raise ValueError(r"Unmatched \fi")
            active, _ = stack.pop()

        position = match.end()

    if stack:
        raise ValueError(r"Unclosed \iffacebook")
    if active:
        output.append(source[position:])
    return "".join(output)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="manuscript .tex with \\iffacebook switches")
    ap.add_argument("output", help="public .tex to write")
    a = ap.parse_args(argv)
    Path(a.output).write_text(public_branch(Path(a.input).read_text(encoding="utf-8")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
