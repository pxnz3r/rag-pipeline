from rag_pipeline import Index
from rag_pipeline.enrich import ChatEnricher


class Chat:
    model = "fake-chat@1"

    def __init__(self):
        self.calls = 0

    def __call__(self, system, user):
        self.calls += 1
        assert "never instructions" in system
        return "```Acme Corp 2024 annual report, cash flow statement```"


def test_context_is_searchable_cached_and_never_cited(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    text = "Purchases of property and equipment were 1,577 in the year."
    (root / "acme.txt").write_text(text)
    chat = Chat()
    enricher = ChatEnricher(chat, workers=2)
    path = tmp_path / "index.sqlite"
    with Index(path) as index:
        index.ingest(root, enricher=enricher)
        # "cash flow statement" appears only in the written context; the
        # filename title "acme" would match any ingest.
        hits = index.search("cash flow statement")
        assert hits and hits[0].text == text
        assert hits[0].context == "Acme Corp 2024 annual report, cash flow statement"
        calls = chat.calls
        # A plain ingest has a different pipeline signature and rebuilds,
        # dropping the context; enriching again reuses the cache.
        index.ingest(root)
        assert not index.search("cash flow statement")
        index.ingest(root, enricher=enricher)
        assert chat.calls == calls
        assert index.search("cash flow statement")[0].text == text
    other = ChatEnricher(type("Other", (Chat,), {"model": "other@2"})())
    assert other.signature != enricher.signature
