"""Lines of code of the *pretokenization* stage, Python implementation vs C++ implementation.

The interviewer asked: "when pretokenization went from Python to C++, did the line count change?"
A whole-file `wc -l` answers a different question (the C++ files also hold encode / decode / merge /
save / load), so this script counts only the functions that belong to the training-time pretokenizer:

  Python  BBPE.pretokenization, process_chunk, find_chunk_boundaries, bytes_2_tuple
  C++     pretokenizeSerial, split_by_special_tokens, simple_split           (pretokenizer.cpp)
          pretokenizeParallel                                               (bpe_tokenizer.cpp)
          find_chunk_boundaries, process_chunk                              (parallel_helper.cpp)
  NOT counted: `text_to_pretokens` (encode path), declarations in headers, the two `print` progress blocks
  are counted (they are part of the function), the regex / split-pattern *definitions* in Python
  `BBPE.__init__` (PAT, SPLIT) are NOT counted -- say so when quoting.

Two numbers per function: `physical` (signature line .. closing line) and `code` (non-blank lines that are
not comment-only and not docstring). The C++ sources carry long teaching comments in Chinese, so quote
`code`, never `physical`.

"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASICS = REPO / "training" / "cs336-basics" / "cs336_basics"

PY_TARGETS = [
    (BASICS / "bpe_tokenizer" / "tokenizerBackEnd.py", "BBPE.pretokenization"),
    (BASICS / "bpe_tokenizer" / "helpers" / "bpe_helper.py", "process_chunk"),
    (BASICS / "bpe_tokenizer" / "helpers" / "bpe_helper.py", "find_chunk_boundaries"),
    (BASICS / "bpe_tokenizer" / "helpers" / "bpe_helper.py", "bytes_2_tuple"),
]
CXX_TARGETS = [  # (file, label, signature regex)
    (BASICS / "bpe_tokenizer_cpp" / "pretokenizer.cpp", "pretokenizeSerial", r"void\s+BPETokenizer::pretokenizeSerial\s*\("),
    (BASICS / "bpe_tokenizer_cpp" / "pretokenizer.cpp", "split_by_special_tokens", r"void\s+BPETokenizer::split_by_special_tokens\s*\("),
    (BASICS / "bpe_tokenizer_cpp" / "pretokenizer.cpp", "simple_split", r"std::vector<std::string>\s+BPETokenizer::simple_split\s*\("),
    (BASICS / "bpe_tokenizer_cpp" / "bpe_tokenizer.cpp", "pretokenizeParallel", r"void\s+BPETokenizer::pretokenizeParallel\s*\("),
    (BASICS / "bpe_tokenizer_cpp" / "parallel_helper.cpp", "find_chunk_boundaries", r"std::vector<size_t>\s+find_chunk_boundaries\s*\("),
    (BASICS / "bpe_tokenizer_cpp" / "parallel_helper.cpp", "process_chunk", r"PretokDict\s+process_chunk\s*\("),
]


# ---------------------------------------------------------------------------------------- Python
def count_py(path: Path, qualname: str) -> dict:
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    parts = qualname.split(".")
    nodes = tree.body
    node = None
    for i, part in enumerate(parts):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise LookupError(f"{qualname} not found in {path}")
        nodes = node.body
    lines = src.splitlines()
    first, last = node.lineno, node.end_lineno
    doc_lines: set[int] = set()
    body0 = node.body[0] if node.body else None
    if isinstance(body0, ast.Expr) and isinstance(getattr(body0, "value", None), ast.Constant) \
            and isinstance(body0.value.value, str):
        doc_lines = set(range(body0.lineno, body0.end_lineno + 1))
    comment_only: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and lines[tok.start[0] - 1].strip().startswith("#"):
            comment_only.add(tok.start[0])
    code = sum(1 for n in range(first, last + 1)
               if lines[n - 1].strip() and n not in doc_lines and n not in comment_only)
    return dict(name=qualname, file=str(path), first=first, last=last, physical=last - first + 1, code=code)


# ---------------------------------------------------------------------------------------- C++
def strip_cxx_comments(src: str) -> str:
    """Blank out // and /* */ comments (string / char literals respected), keeping every newline."""
    out, i, n = [], 0, len(src)
    while i < n:
        c, d = src[i], src[i + 1] if i + 1 < n else ""
        if c == "/" and d == "/":
            while i < n and src[i] != "\n":
                i += 1
        elif c == "/" and d == "*":
            i += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            i += 2
        elif c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n and src[i] != q:  # literal contents are blanked so braces inside them cannot count
                if src[i] == "\\" and i + 1 < n:
                    out.append(" ")
                    i += 1
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            out.append(q)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def count_cxx(path: Path, label: str, signature_regex: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    clean = strip_cxx_comments(raw)
    m = re.search(signature_regex, clean)
    if not m:
        raise LookupError(f"{signature_regex!r} not found in {path}")
    open_at = clean.index("{", m.end())
    depth, j = 0, open_at
    while True:
        if clean[j] == "{":
            depth += 1
        elif clean[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    first = clean.count("\n", 0, m.start()) + 1
    last = clean.count("\n", 0, j) + 1
    clean_lines = clean.splitlines()
    code = sum(1 for n in range(first, last + 1) if clean_lines[n - 1].strip())
    return dict(name=label, file=str(path), first=first, last=last, physical=last - first + 1, code=code)


def pretok_loc() -> dict:
    py = [count_py(p, q) for p, q in PY_TARGETS]
    cx = [count_cxx(p, label, sig) for p, label, sig in CXX_TARGETS]
    return dict(
        python=dict(functions=py, code=sum(f["code"] for f in py), physical=sum(f["physical"] for f in py)),
        cxx=dict(functions=cx, code=sum(f["code"] for f in cx), physical=sum(f["physical"] for f in cx)),
    )


if __name__ == "__main__":
    r = pretok_loc()
    for lang in ("python", "cxx"):
        print(f"== {lang}")
        for f in r[lang]["functions"]:
            print(f"  {f['name']:28s} lines {f['first']:4d}-{f['last']:4d}  physical={f['physical']:4d}  code={f['code']:4d}")
        print(f"  TOTAL physical={r[lang]['physical']}  code={r[lang]['code']}")
    print(f"code-line ratio C++/Python = {r['cxx']['code'] / r['python']['code']:.2f}")
