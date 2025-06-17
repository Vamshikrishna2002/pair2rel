__version__ = "1.1.1.dev"

from .model import Pair2Rel
from typing import Optional, Union, List
import torch

__all__ = ["Pair2Rel"]


# https://github.com/tomaarsen/SpanMarkerNER/blob/main/span_marker/__init__.py
# Set up for spaCy
try:
    from spacy.language import Language
except ImportError:
    pass
else:

    DEFAULT_SPACY_CONFIG = {
        "model": "chapalavamshi022/pair2rel",
        "batch_size": 1,
        "device": None,
        "threshold": 0.3,
    }

    @Language.factory(
        "pair2rel",
        assigns=["doc._.relations"],
        default_config=DEFAULT_SPACY_CONFIG,
    )
    def _spacy_pair2rel_factory(
        nlp: Language,
        name: str, 
        model: str,
        batch_size: int,
        device: Optional[Union[str, torch.device]],
        threshold: float,
    ) -> "SpacyPair2RelWrapper":
        from pair2rel.spacy_integration import SpacyPair2RelWrapper
        return SpacyPair2RelWrapper(model, batch_size=batch_size, device=device)