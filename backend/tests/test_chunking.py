import pytest
from app.chunking import chunk_pages
from app.pdf import PageText, TextUnit


def page(n, *units):
    us = [TextUnit(k, t) for k, t in units]
    return PageText(n, "\n".join(u.text for u in us), us)


SENT = "The office opened in 2019 and employs staff in several departments."


def test_chunks_never_cross_pages_and_keep_their_page():
    pages = [page(1, ("para", SENT * 3)), page(2, ("para", "Page two only. " + SENT * 3))]
    chunks = chunk_pages(pages, 25, 5)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.page == 2 for c in chunks if "Page two only" in c.text)
    assert all("Page two only" not in c.text for c in chunks if c.page == 1)
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_chunks_end_on_sentence_boundaries_with_sentence_overlap():
    text = " ".join(f"Sentence number {i} says something useful about topic {i}." for i in range(1, 13))
    chunks = chunk_pages([page(1, ("para", text))], 30, 12)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.endswith(".") and c.text.startswith("Sentence")
    first_last = chunks[0].text.split(". ")[-1]
    assert first_last.rstrip(".") in chunks[1].text  # last sentence of chunk 1 repeats at the start of chunk 2


def test_section_follows_headings_and_carries_across_pages():
    pages = [page(1, ("heading", "Financial Results"), ("para", SENT)), page(2, ("para", "Costs were stable.")), page(3, ("heading", "Outlook"), ("para", "Growth expected."))]
    chunks = chunk_pages(pages, 40, 5)
    assert [(c.page, c.section) for c in chunks] == [(1, "Financial Results"), (2, "Financial Results"), (3, "Outlook")]


def test_heading_is_not_stranded_at_the_end_of_a_long_chunk():
    long = " ".join(f"Filler sentence {i} about routine operations." for i in range(8))
    chunks = chunk_pages([page(1, ("para", long), ("heading", "Next Topic"), ("para", "Details of the next topic follow here."))], 40, 5)
    assert not any(c.text.rstrip().endswith("Next Topic") for c in chunks)
    assert any("Next Topic\nDetails of the next topic" in c.text for c in chunks)  # heading stays with its paragraph


def test_table_rows_get_their_own_chunks_flagged_as_table():
    chunks = chunk_pages([page(1, ("para", SENT), ("table", "Europe: 2022 is 38.2; 2023 is 41.5."), ("table", "Asia: 2022 is 21.0; 2023 is 24.8."), ("para", "After the table."))], 60, 5)
    kinds = [c.kind for c in chunks]
    assert kinds == ["text", "table", "text"]
    assert "Europe" in chunks[1].text and "Asia" in chunks[1].text


def test_oversized_sentence_is_split_not_dropped():
    chunks = chunk_pages([page(1, ("para", "word " * 500))], 50, 5)
    assert sum(len(c.text.split()) for c in chunks) >= 500


def test_plain_text_pages_without_units_still_work():
    chunks = chunk_pages([PageText(1, "Annual Report\nThe company opened its Lisbon office in 2019.")], 40, 5)
    assert chunks[0].section == "Annual Report"


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([page(1, ("para", SENT))], size, overlap)
