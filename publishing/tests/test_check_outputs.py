"""What the output check counts as a fingerprint in a Word file."""

import zipfile

from check_outputs import zip_findings

BULLETS = ('<w:numbering><w:lvl w:ilvl="0"><w:lvlText w:val="\uf0b7"/></w:lvl>'
           '<w:lvl w:ilvl="1"><w:lvlText w:val="\uf0a7"/></w:lvl></w:numbering>')


def docx(tmp_path, entries):
    path = tmp_path / "x.docx"
    with zipfile.ZipFile(path, "w") as z:
        for name, text in entries.items():
            z.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), text)
    return path


def test_word_bullet_glyphs_are_not_hidden_characters(tmp_path):
    # the bullets of a list are Symbol and Wingdings font slots, not text
    assert zip_findings(docx(tmp_path, {"word/numbering.xml": BULLETS}), "x") == []


def test_a_private_use_character_in_the_text_is_still_found(tmp_path):
    body = "<w:document><w:t>a\uf0b7b</w:t></w:document>"
    found = zip_findings(docx(tmp_path, {"word/document.xml": body}), "x")
    assert found == ["word/document.xml: invisible U+F0B7"]
