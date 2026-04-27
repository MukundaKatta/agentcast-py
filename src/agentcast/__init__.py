"""agentcast -- structured output for any LLM call.

Public surface (mirrors the JS sibling, with async-by-default for the LLM
call since that's the natural Python pattern):

* ``cast(...)`` -- async helper that calls your LLM, extracts JSON, validates,
  retries with the validation error as feedback. Throws :class:`CastError`
  if all retries fail.
* ``extract_json(text)`` -- pull JSON out of an LLM response. Sync.
* ``adapters`` -- bridges for shape-spec, predicate, and pydantic validators.
* ``CastError`` -- raised when retries are exhausted.
"""

from .adapters import adapters
from .cast import Attempt, cast
from .errors import CastError
from .extract import extract_json

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "Attempt",
    "CastError",
    "adapters",
    "cast",
    "extract_json",
]
