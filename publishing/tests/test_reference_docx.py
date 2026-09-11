"""The Word reference document is pandoc's own with the links in a darker blue."""

import re
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import reference_docx  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None, reason="needs pandoc")


def styles_of(docx: bytes) -> str:
    return zipfile.ZipFile(BytesIO(docx)).read("word/styles.xml").decode()


def test_the_hyperlink_style_is_recoloured_and_loses_its_theme_colour():
    out = styles_of(reference_docx.restyle(reference_docx.pandoc_default()))
    block = reference_docx.HYPERLINK_STYLE.search(out).group(0)
    assert f'<w:color w:val="{reference_docx.LINK_COLOR}"/>' in block
    assert "themeColor" not in block


def test_every_other_style_is_left_as_pandoc_ships_it():
    before = styles_of(reference_docx.pandoc_default())
    after = styles_of(reference_docx.restyle(reference_docx.pandoc_default()))
    strip = lambda s: reference_docx.HYPERLINK_STYLE.sub("", s)  # noqa: E731
    assert strip(before) == strip(after)


def test_a_reference_without_a_hyperlink_style_is_refused():
    default = reference_docx.pandoc_default()
    src = zipfile.ZipFile(BytesIO(default))
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/styles.xml":
                data = reference_docx.HYPERLINK_STYLE.sub("", data.decode()).encode()
            dst.writestr(item, data)
    with pytest.raises(SystemExit, match="Hyperlink"):
        reference_docx.restyle(out.getvalue())
