import pytest
from app.chunking import chunk_pages
from app.config import Settings
from app.llm import LLMConfigError, get_llm, validate_llm_config
from app.pdf import ExtractionError, PDFValidationError, extract_pages, validate_pdf
from app.retrieval import rrf_fuse
from app.security import RateLimiter, create_token, hash_password, safe_display_name, verify_password, verify_token
from conftest import REPORT_PAGES, make_pdf


# ---- PDF validation / extraction ----
def test_validate_accepts_real_pdf():
    assert validate_pdf(make_pdf(REPORT_PAGES), 10**7, 50) == 3


@pytest.mark.parametrize("data,msg", [(b"", "empty"), (b"hello world not a pdf", "not a PDF"), (b"%PDF-1.4 garbage", "corrupted")])
def test_validate_rejects_bad_content(data, msg):
    with pytest.raises(PDFValidationError, match=msg):
        validate_pdf(data, 10**7, 50)


def test_validate_rejects_oversize_and_too_many_pages():
    pdf = make_pdf(REPORT_PAGES)
    with pytest.raises(PDFValidationError, match="larger"):
        validate_pdf(pdf, 100, 50)
    with pytest.raises(PDFValidationError, match="pages"):
        validate_pdf(pdf, 10**7, 2)


def test_extraction_preserves_page_numbers():
    pages, method = extract_pages(make_pdf(REPORT_PAGES), 20)
    assert method == "pymupdf" and [p.page for p in pages] == [1, 2, 3]
    assert "Lisbon" in pages[0].text and "45 million" in pages[1].text


def test_extraction_reports_empty_pdf():
    with pytest.raises(ExtractionError, match="No usable text"):
        extract_pages(make_pdf(["", ""]), 20)


# ---- chunking ----
def test_chunks_keep_page_and_never_cross_pages():
    pages, _ = extract_pages(make_pdf(["word " * 100 + "alpha", "beta " * 30]), 5)
    chunks = chunk_pages(pages, 40, 5)
    assert len(chunks) >= 3
    assert {c.page for c in chunks} == {1, 2}
    assert all("beta" not in c.text for c in chunks if c.page == 1)
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_chunk_params_validated():
    with pytest.raises(ValueError):
        chunk_pages([], 10, 10)


def test_chunk_section_detected():
    pages, _ = extract_pages(make_pdf(REPORT_PAGES), 20)
    chunks = chunk_pages(pages, 40, 5)
    assert next(c for c in chunks if c.page == 2).section == "Financial Results"


# ---- retrieval math / claims / numeric ----
def test_rrf_prefers_items_ranked_high_in_both_lists():
    scores = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"]) and scores["a"] > scores["c"] > scores["d"] - 1e-9 or scores["c"] > scores["d"]
    assert max(scores, key=scores.get) in {"a", "b"}


# ---- security helpers / config ----
def test_password_and_token_roundtrip():
    h = hash_password("s3cret-pass-word")
    assert verify_password("s3cret-pass-word", h) and not verify_password("wrong", h)
    tok = create_token("u1", "k" * 40, 5)
    assert verify_token(tok, "k" * 40) == "u1"
    assert verify_token(tok, "z" * 40) is None
    assert verify_token(tok[:-2] + "xx", "k" * 40) is None
    assert verify_token(create_token("u1", "k" * 40, -1), "k" * 40) is None


def test_safe_display_name_strips_paths():
    assert safe_display_name("../../etc/passwd\x00.pdf") == "passwd.pdf"
    assert safe_display_name("C:\\x\\y\\file.pdf") == "file.pdf"


def test_rate_limiter():
    rl = RateLimiter(2)
    assert rl.allow("k") and rl.allow("k") and not rl.allow("k") and rl.allow("other")


def test_llm_config_validation(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert "LLM_PROVIDER" in validate_llm_config(s)[0]
    with pytest.raises(LLMConfigError):
        get_llm(s)
    assert validate_llm_config(Settings(data_dir=tmp_path, llm_provider="openai", llm_model="m"))[0].startswith("LLM_API_KEY")
    assert validate_llm_config(Settings(data_dir=tmp_path, llm_provider="ollama", llm_model="m")) == []
    assert "not supported" in validate_llm_config(Settings(data_dir=tmp_path, llm_provider="nope"))[0]


def test_production_requires_secret(tmp_path):
    with pytest.raises(RuntimeError):
        Settings(data_dir=tmp_path, app_env="production").resolve_secret_key()
    assert Settings(data_dir=tmp_path).resolve_secret_key()  # dev: generated and persisted
    assert Settings(max_retries=9).effective_max_retries == 2
