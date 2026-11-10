"""Entry point for the scorer: build() returns the knowledge system. See README.md for the design."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:                                    # the harness puts its own folder on sys.path
    from contract import Answer, Conflict, Source
except ImportError:                     # running outside the harness: same shapes, defined locally
    @dataclass
    class Source:                       # type: ignore[no-redef]
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:                       # type: ignore[no-redef]
        text: str
        sources: List[Source] = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:                     # type: ignore[no-redef]
        topic: str
        values: List[str]
        sources: List[Source]
        resolution: Optional[str] = None
        reason: Optional[str] = None

def _load_env_file() -> None:
    """Local convenience: read the API key from a .env file in this folder or the one above it
    (and this system's own KB_ settings from the one in this folder), without overriding anything
    already set in the environment. Nothing else in those files is touched. Graders can simply
    export the key."""
    here = Path(__file__).resolve().parent
    for candidate, settings_too in ((here / ".env", True), (here.parent / ".env", False)):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            name = name.replace("export ", "").strip()
            wanted = name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") or (settings_too and name.startswith("KB_"))
            if sep and wanted and value.strip() and name not in os.environ:
                os.environ[name] = value.strip().strip("'\"")


_load_env_file()

from kb.system import KnowledgeSystem  # noqa: E402


def build() -> KnowledgeSystem:
    return KnowledgeSystem(answer_types=(Answer, Source, Conflict))
