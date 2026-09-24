import os
import tarfile

from archive import pack


def test_the_same_files_give_the_same_archive_whoever_packs_them_and_whenever(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")
    src = tmp_path / "bundle"
    (src / "sub").mkdir(parents=True)
    (src / "a.tex").write_text("text")
    (src / "sub" / "b.sty").write_text("style")
    pack(src, tmp_path / "one.tar.gz")
    for f in (src / "a.tex", src / "sub" / "b.sty"):
        os.utime(f, (1, 1))                       # touched since: a later build of the same files
    pack(src, tmp_path / "two.tar.gz")
    assert (tmp_path / "one.tar.gz").read_bytes() == (tmp_path / "two.tar.gz").read_bytes()
    raw = (tmp_path / "one.tar.gz").read_bytes()
    assert raw[4:8] == b"\0\0\0\0" and not raw[3] & 0x08       # gzip: no time, no file name
    with tarfile.open(tmp_path / "one.tar.gz") as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == ["bundle", "bundle/a.tex", "bundle/sub", "bundle/sub/b.sty"]
        assert {(m.uid, m.gid, m.uname, m.gname, m.mtime) for m in members} == {(0, 0, "", "", 1767225600)}
