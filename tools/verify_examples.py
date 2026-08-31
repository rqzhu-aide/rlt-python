#!/usr/bin/env python
"""Extract every fenced ```python block from docs/ pages and execute it.

Each page's blocks are concatenated into one module (matching how a reader
would run them top-to-bottom), executed with matplotlib Agg backend.
Blocks are dedented (mkdocs tabbed content indents fences), and bare
trailing expressions are displayed like a notebook would. Open figures are
saved to build/verify_figs/ after each page. Exit 1 on any failure.
"""
import ast
import io
import os
import re
import sys
import textwrap
import traceback
from contextlib import redirect_stdout

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DOCS = os.path.join(ROOT, "docs")
FIGS = os.path.join(ROOT, "build", "verify_figs")
os.makedirs(FIGS, exist_ok=True)

page_re = re.compile(r"^(?: {0,8})```python\n(.*?)^(?: {0,8})```", re.DOTALL | re.MULTILINE)


def transform(code):
    """Turn top-level bare expressions into __display(...) calls."""
    tree = ast.parse(code)
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            call = ast.Call(func=ast.Name(id="__display", ctx=ast.Load()),
                            args=[node.value], keywords=[])
            new_body.append(ast.Expr(value=call))
        else:
            new_body.append(node)
    tree.body = new_body
    ast.fix_missing_locations(tree)
    return compile(tree, "<doc>", "exec")


def display(v):
    print(repr(v))


pages = []
for root, _dirs, files in os.walk(DOCS):
    for f in sorted(files):
        if f.endswith(".md"):
            pages.append(os.path.join(root, f))
pages.sort()

total_blocks = 0
failures = []
fig_count = 0
records = []

for path in pages:
    rel = os.path.relpath(path, ROOT)
    text = open(path).read()
    blocks = [textwrap.dedent(b) for b in page_re.findall(text)]
    if not blocks:
        print(f"[skip] {rel}: no python blocks")
        continue
    code = "\n\n".join(blocks)
    nb = len(blocks)
    total_blocks += nb
    buf = io.StringIO()
    print(f"[run ] {rel}: {nb} block(s), {len(code)} chars")
    try:
        with redirect_stdout(buf):
            exec(transform(code), {"__name__": "__main__",
                                   "__file__": path,
                                   "__display": display})
    except Exception:
        failures.append(rel)
        print(f"  !! FAILED: {rel}")
        tb = traceback.format_exc(limit=6)
        print("  " + tb.replace("\n", "\n  ")[:2000])
    # save any open figures from this page
    for num in plt.get_fignums():
        fig_count += 1
        fig = plt.figure(num)
        fig.savefig(os.path.join(FIGS, f"{os.path.basename(rel)[:-3]}_{fig_count:02d}.png"),
                    dpi=80)
    plt.close("all")
    out = buf.getvalue()
    if out.strip():
        records.append((rel, out))
        print("  ---- stdout ----")
        for line in out.splitlines():
            print(f"  | {line}")

print()
print("=" * 60)
print(f"total python blocks executed: {total_blocks}")
print(f"figures rendered: {fig_count}")
print(f"failures: {len(failures)} {failures}")

with open(os.path.join(ROOT, "build", "verify_stdout.txt"), "w") as fh:
    for rel, out in records:
        fh.write(f"##### {rel}\n{out}\n")
sys.exit(1 if failures else 0)
