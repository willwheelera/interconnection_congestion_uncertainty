"""Repository locations, resolved once.

Every module that needs a repository-relative path imports it from here rather than
counting `Path(__file__).parents[...]` levels of its own.  Counting levels breaks
silently whenever a file moves between directories, which is exactly what the
stage-based layout does.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
