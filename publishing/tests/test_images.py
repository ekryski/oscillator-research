"""Pointing the LaTeX builds at the vector copy of each figure."""

from preprocess import to_vector_images


def test_a_local_png_becomes_the_pdf_beside_it():
    out, n = to_vector_images("![A figure.](resources/figures/fig7-model-timeline.png)")
    assert out == "![A figure.](resources/figures/fig7-model-timeline.pdf)"
    assert n == 1


def test_a_caption_wrapped_over_lines_is_still_matched():
    # every caption in these manuscripts wraps; a line-anchored pattern misses them
    src = "![A caption that\nruns over two lines.](resources/figures/g0-pipeline-order.png)"
    out, n = to_vector_images(src)
    assert out.endswith("g0-pipeline-order.pdf)")
    assert n == 1


def test_a_remote_image_is_left_alone():
    # there is no PDF beside a URL to swap to
    src = "![Remote.](https://example.org/x.png)"
    assert to_vector_images(src) == (src, 0)


def test_a_link_is_not_an_image():
    src = "[a png file](resources/figures/fig7-model-timeline.png)"
    assert to_vector_images(src) == (src, 0)


def test_several_figures_in_one_document():
    src = ("![One.](resources/figures/a.png)\n\ntext\n\n![Two.](resources/figures/b.png)")
    out, n = to_vector_images(src)
    assert n == 2 and ".png" not in out


def test_a_non_png_image_is_left_alone():
    src = "![Vector already.](resources/figures/a.pdf)"
    assert to_vector_images(src) == (src, 0)


def test_a_caption_containing_a_citation_link_still_converts():
    # the nested [] used to defeat the caption pattern, leaving the image on .png
    src = ("![Controls, per [Muzellec et al. 2025](https://arxiv.org/abs/2502.21077)."
           "](resources/figures/fig13-controls-grid.png)")
    out, n = to_vector_images(src)
    assert n == 1 and out.endswith("fig13-controls-grid.pdf)")


def test_zero_width_and_tag_characters_are_stripped():
    from preprocess import strip_hidden
    # a tag-block run is the usual way generated prose is watermarked
    payload = "".join(chr(0xE0000 + ord(c) - 0x20) for c in "ID")
    out, n = strip_hidden(f"real​text{payload}️⁠ here")
    assert out == "realtext here" and n == 5


def test_a_no_break_space_becomes_a_space_rather_than_vanishing():
    from preprocess import strip_hidden
    out, n = strip_hidden("two words")
    assert out == "two words" and n == 1


def test_ordinary_text_is_untouched():
    from preprocess import strip_hidden
    src = "Kuramoto–Sakaguchi, Buzsáki, θ and ω, 0.97–1.00."
    assert strip_hidden(src) == (src, 0)
