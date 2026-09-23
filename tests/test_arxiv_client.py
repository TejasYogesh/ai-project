# tests/test_arxiv_client.py
from backend.services.arxiv_client import _build_search_query, extract_arxiv_id


def test_extract_arxiv_id_from_url():
    assert extract_arxiv_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"


def test_stop_words_are_dropped():
    assert _build_search_query("Attention Is All You Need") == \
        "all:Attention AND all:All AND all:You AND all:Need"


def test_field_prefix_passes_through():
    assert _build_search_query('ti:"Attention Is All You Need"') == 'ti:"Attention Is All You Need"'