"""Inline outputs/dashboard_data.json into the dashboard template.

Writes two files:
  dashboard/index.html          full HTML document (open locally or host on GitHub Pages)
  outputs/fly-connectome-atlas.html   body-only page, for hosts that add their own <head>
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / 'dashboard' / 'template.html'
DATA = ROOT / 'outputs' / 'dashboard_data.json'

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""


def main():
    page = TEMPLATE.read_text().replace('/*__DATA__*/null', DATA.read_text())
    # Split the template's leading <title>/<meta>/<link>/<style> block into <head>.
    head_end = page.index('</style>') + len('</style>')
    full = HEAD + page[:head_end] + '\n</head>\n<body>\n' + page[head_end:] + '\n</body>\n</html>\n'
    (ROOT / 'dashboard' / 'index.html').write_text(full)
    (ROOT / 'outputs' / 'fly-connectome-atlas.html').write_text(page)
    print(f'dashboard/index.html  {len(full) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
