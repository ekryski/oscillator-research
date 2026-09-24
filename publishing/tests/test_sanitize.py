from sanitize import clean


def test_a_tag_block_watermark_and_zero_width_marks_are_deleted():
    payload = "".join(chr(0xE0000 + ord(c)) for c in "model-v1")
    out, found = clean(f"The re​sult{payload} holds⁣.")
    assert out == "The result holds." and found == {"invisible": 10}


def test_control_characters_go_but_tab_and_newline_stay():
    out, found = clean("a\x00b\x1bc\x7fd\x85e\tf\ng")
    assert out == "abcde\tf\ng" and found == {"invisible": 4}


def test_unusual_spaces_become_ordinary_spaces():
    out, found = clean("10 dB and 3 points　here now")
    assert out == "10 dB and 3 points here now" and found == {"spaces": 4}


def test_a_cyrillic_letter_inside_a_latin_word_becomes_latin():
    out, found = clean("Kurаmoto and Сoupling")       # Cyrillic а, Cyrillic С
    assert out == "Kuramoto and Coupling" and found == {"look-alikes": 2}


def test_greek_words_and_symbols_are_left_alone():
    src = "θ̇ᵢ = ωᵢ, λ = 0.3, Kuramoto–Sakaguchi, Buzsáki, Ο(n)"
    assert clean(src) == (src, {})


def test_fullwidth_ascii_becomes_ascii():
    out, found = clean("ＡＢＣ１２")
    assert out == "ABC12" and found == {"look-alikes": 5}


def test_trailing_whitespace_goes_and_a_markdown_hard_break_keeps_its_meaning():
    assert clean("one \ntwo\t\nthree ") == ("one\ntwo\nthree", {"trailing": 3})
    assert clean("line one  \nline two", markdown=True) == ("line one\\\nline two", {"trailing": 1})


def _chunk(cid: bytes, body: bytes) -> bytes:
    return cid + len(body).to_bytes(4, "little") + body + (b"\0" if len(body) % 2 else b"")


def test_a_wav_keeps_its_samples_and_loses_the_peak_chunk_that_dates_it():
    from sanitize import strip_wav
    fmt, data = _chunk(b"fmt ", bytes(16)), _chunk(b"data", bytes(range(8)))
    body = fmt + _chunk(b"PEAK", bytes(16)) + _chunk(b"LIST", b"INFOISFTx") + data
    raw = b"RIFF" + (4 + len(body)).to_bytes(4, "little") + b"WAVE" + body
    out, n = strip_wav(raw)
    assert n == 2 and out == b"RIFF" + (4 + len(fmt + data)).to_bytes(4, "little") + b"WAVE" + fmt + data


def test_a_png_loses_its_text_and_time_chunks():
    import zlib
    from sanitize import strip_png

    def chunk(cid: bytes, body: bytes) -> bytes:
        return len(body).to_bytes(4, "big") + cid + body + zlib.crc32(cid + body).to_bytes(4, "big")
    head, tail = chunk(b"IHDR", bytes(13)), chunk(b"IDAT", b"x") + chunk(b"IEND", b"")
    raw = b"\x89PNG\r\n\x1a\n" + head + chunk(b"tEXt", b"Software\0tool 1.0") + chunk(b"tIME", bytes(7)) + tail
    assert strip_png(raw) == (b"\x89PNG\r\n\x1a\n" + head + tail, 2)
