"""Convert a markdown file to a standalone styled HTML file.

Usage:
    python scripts/md_to_html.py path/to/file.md [path/to/out.html]

If out path is omitted, writes alongside the input with .html extension.
"""
import sys
from pathlib import Path

import markdown

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.6; color: #24292f; }
h1, h2, h3, h4 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; }
code { background: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-size: 95%; }
pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #d0d7de; padding: 6px 12px; }
th { background: #f6f8fa; }
blockquote { border-left: 4px solid #d0d7de; padding: 0 1em; color: #57606a; margin-left: 0; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #eaecef; }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert(md_path: Path, html_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    # Rewrite relative .md links to .html so cross-doc links keep working
    body = body.replace('.md"', '.html"').replace(".md#", ".html#")
    title = md_path.stem
    html_path.write_text(
        TEMPLATE.format(title=title, css=CSS, body=body),
        encoding="utf-8",
    )
    print(f"{md_path} -> {html_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    md_path = Path(sys.argv[1])
    html_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else md_path.with_suffix(".html")
    convert(md_path, html_path)


if __name__ == "__main__":
    main()
