"""Domain packs: what changes per field, bundled so a domain needs no fork.

A pack sets search defaults, adds answer guidance to the generation prompt,
names the metadata its documents should carry, and can keep every stage local.
Packs layer under a ``--config`` file and over the tier profile or preset:

    rag-pipeline --profile auto --domain trading ask "Guidance change?" --as-of 2024-07-21

Third-party packs register through the ``rag_pipeline.domains`` entry-point
group (a callable returning a list of ``Domain``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    name: str
    guidance: str
    metadata: tuple[str, ...] = ()
    search: dict = field(default_factory=dict)
    # Keep every stage on this machine unless RAG_ALLOW_NETWORK=1 is set.
    local_only: bool = False


BUILTIN = [
    Domain(
        "finance",
        "Financial filings: state the fiscal period and unit of every figure, "
        "prefer audited statements over narrative sections, and use the "
        "calculator for any ratio, margin or growth rate.",
        ("company", "fiscal_year", "period", "doc_type", "published"),
    ),
    Domain(
        "law",
        "Legal documents: quote the operative clause verbatim, name the "
        "governing law and jurisdiction when stated, and distinguish binding "
        "text from recitals, definitions and headings.",
        ("jurisdiction", "contract", "party", "effective", "published"),
        {"match": "any"},
    ),
    Domain(
        "healthcare",
        "Clinical and biomedical sources: answer only what the cited text "
        "states, say when evidence is preliminary or conflicting, and abstain "
        "rather than extrapolate to a patient or dosage.",
        ("source", "study_type", "published"),
        local_only=True,
    ),
    Domain(
        "trading",
        "Market documents: give the publication time of each source, never use "
        "information dated after the question's as-of time, and separate "
        "reported figures from guidance or estimates.",
        ("ticker", "published", "source"),
    ),
    Domain("general", "", ()),
]


def registry():
    from importlib.metadata import entry_points

    found = {d.name: d for d in BUILTIN}
    for point in sorted(
        entry_points(group="rag_pipeline.domains"), key=lambda p: p.name
    ):
        for domain in point.load()():
            if not isinstance(domain, Domain) or domain.name in found:
                raise ValueError(
                    f"Plugin {point.name} returned an invalid or duplicate domain"
                )
            found[domain.name] = domain
    return found


def get(name: str) -> Domain:
    domains = registry()
    if name not in domains:
        raise ValueError(f"Unknown domain {name!r}; choose from {sorted(domains)}")
    return domains[name]


# Legal citation patterns: U.S. Code, C.F.R., reporters, and bare sections.
CITATION = re.compile(
    r"\b\d+\s+U\.S\.C\.?\s*§+\s*[\w.()-]+"
    r"|\b\d+\s+C\.F\.R\.?\s*§*\s*\d[\d.]*[\w-]*"
    r"|\b\d+\s+(?:U\.S\.|S\.\s?Ct\.|L\.\s?Ed\.(?:\s?2d)?|F\.(?:\s?Supp\.)?(?:\s?[234]d|\s?4th)?)\s+\d+"
    r"|§+\s*\d+[\w.()-]*"
)


def citations(text: str):
    """Legal citations with exact offsets into ``text`` (derived, checkable)."""
    return [
        {
            "citation": m.group().rstrip(".,;"),
            "start": m.start(),
            "end": m.start() + len(m.group().rstrip(".,;")),
        }
        for m in CITATION.finditer(text)
    ]
