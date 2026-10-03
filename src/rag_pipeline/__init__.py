"""Evidence-first local retrieval. Public APIs are loaded lazily for PDF workers."""

__version__ = "0.4.0"
__all__ = [
    "Index",
    "Hit",
    "Answer",
    "Operand",
    "answer",
    "calculate",
    "evaluate_rankings",
    "Navigation",
]


def __getattr__(name):
    if name == "Navigation":
        from .navigation import Navigation

        return Navigation
    if name in {"Index", "Hit"}:
        from . import index

        return getattr(index, name)
    if name in {"Answer", "Operand", "answer", "calculate"}:
        from . import answers

        return getattr(answers, name)
    if name == "evaluate_rankings":
        from .retrieval_metrics import evaluate_rankings

        return evaluate_rankings
    raise AttributeError(name)
