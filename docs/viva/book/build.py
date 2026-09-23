"""Assemble the project book and print it to PDF with headless Chrome.

    python docs/viva/book/build.py            -> docs/viva/project-book.pdf

Chapters are the numbered *.html files in this directory, concatenated in name
order. Any <h1> carrying data-toc="..." becomes a contents entry (data-num adds
a chapter number); data-part="..." on an <h1> starts a part heading in the
contents. Chrome turns the headings into PDF bookmarks.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "project-book.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

HEADING = re.compile(r'<h1 id="(?P<id>[^"]+)"(?P<attrs>[^>]*)>')
ATTR = re.compile(r'data-(toc|num|part)="([^"]*)"')

COVER = """
<section class="cover">
  <div class="eyebrow">MLOPS · CHURN PREDICTION · REAL DATA</div>
  <h1>Subscriber Dropout Detection</h1>
  <p class="sub">The complete project, end to end, and a viva question bank &mdash;
  every number in this book comes from the running system.</p>
  <div class="stats">
    <div><b>82.8M</b><span>events in a Postgres warehouse</span></div>
    <div><b>8</b><span>services, one command</span></div>
    <div><b>441</b><span>tests &middot; 7 CI/CD jobs</span></div>
  </div>
  <div class="foot">github.com/markandeyavarma3-lab/subscriber-dropout-detection</div>
</section>
"""


def contents(body: str) -> str:
    items = []
    for match in HEADING.finditer(body):
        attrs = dict(ATTR.findall(match.group("attrs")))
        if "part" in attrs:
            items.append(f'<li class="part">{html.escape(attrs["part"])}</li>')
        if "toc" in attrs:
            num = f'<span class="n">{html.escape(attrs.get("num", ""))}</span>'
            items.append(
                f'<li class="ch">{num}<a href="#{match.group("id")}">'
                f'{html.escape(attrs["toc"])}</a></li>'
            )
    return ('<section class="toc"><h1 id="contents">Contents</h1><ol>'
            + "".join(items) + "</ol></section>")


def main() -> int:
    chapters = sorted(HERE.glob("[0-9][0-9]-*.html"))
    body = "\n".join(path.read_text() for path in chapters)
    # Images are referenced from the repo root, so one path works for both
    # the HTML preview and the print.
    body = body.replace('src="/', f'src="file://{ROOT}/')
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Subscriber Dropout Detection - Project Book</title>"
        f"<style>{(HERE / 'style.css').read_text()}</style></head><body>"
        + COVER + contents(body) + body + "</body></html>"
    )
    target = HERE / "book.html"
    target.write_text(page)

    result = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--generate-pdf-document-outline", "--allow-file-access-from-files",
         f"--print-to-pdf={OUT}", "--virtual-time-budget=10000", f"file://{target}"],
        capture_output=True, text=True,
    )
    if not OUT.exists():
        print(result.stderr[-2000:], file=sys.stderr)
        return 1
    print(f"wrote {OUT.relative_to(ROOT)} from {len(chapters)} chapter files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
