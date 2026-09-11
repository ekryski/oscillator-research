#!/usr/bin/env python3
"""Print the author list as a plain display string.

pandoc's stock LaTeX template emits `$author$` directly into `\\author{}`. Our
metadata makes each author a map — name, affiliation, email — because the TMLR
template needs the parts separately, and pandoc stringifies a map as the word
"true". So the reading formats are handed a display string built from the same
metadata rather than a second, drifting copy of the author list.

    python3 publishing/lib/byline.py papers/<paper>/metadata/paper.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_meta import authors  # noqa: E402

print(" and ".join(a["name"] for a in authors(Path(sys.argv[1]).read_text())))
