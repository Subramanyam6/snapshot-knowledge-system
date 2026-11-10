"""Search by meaning, as a complement to keyword search.

Keyword search finds a page only when it shares words with the question. "How do I change my phone
number?" shares none with a page that says "update your personal information". A small embedding
model closes that gap: it turns a text into a list of numbers (a vector) such that texts with a
similar meaning get similar vectors, whatever words they use.

The model runs on this machine: no API call, no per-question cost, and the website text never
leaves it. It is optional. When the library or the model files cannot be loaded, the system
searches by keywords alone, exactly as it does without this module.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

MODEL_NAME = os.environ.get("KB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TEXT_CHARS = 1200        # roughly what the model reads (512 word pieces); it cuts longer text anyway
BATCH = 16               # measured: small batches of similar-length texts are fastest on a CPU
CHUNK = 256              # texts embedded between two looks at the clock
# Embedding every passage is the slow part of loading a snapshot (about three minutes in total on a
# laptop, once). It must never endanger the run, so it works to a time budget: when one snapshot
# takes too long, that snapshot is searched by keywords only and the work done so far is kept. The
# total (20 minutes) leaves room inside the scorer's 30-minute loading limit on a machine several
# times slower than a laptop.
SECONDS_PER_SNAPSHOT = float(os.environ.get("KB_EMBED_SECONDS", "400"))
SECONDS_IN_TOTAL = 3 * SECONDS_PER_SNAPSHOT


def cache_home() -> Path:
    """Where the model files and saved vectors go: KB_CACHE_DIR if set, else a .cache folder next
    to solution.py, else (a read-only checkout) a folder in the system's temporary directory."""
    if os.environ.get("KB_CACHE_DIR"):
        return Path(os.environ["KB_CACHE_DIR"])
    home = Path(__file__).resolve().parents[1] / ".cache"
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".writable"
        probe.write_text("")
        probe.unlink()
        return home
    except OSError:
        return Path(tempfile.gettempdir()) / "kb-cache"


def _digest(text: str) -> str:
    return hashlib.sha1((MODEL_NAME + "\n" + text).encode("utf-8")).hexdigest()


class Embedder:
    """Turns passages and questions into unit-length vectors, remembering every passage it has
    already seen: most pages do not change from one snapshot to the next."""

    def __init__(self):
        # Keep the model files and the hub's bookkeeping inside our own folder, and send neither an
        # account token nor usage statistics with the download: the model is public.
        home = cache_home()
        os.environ.setdefault("HF_HOME", str(home / "huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        import numpy
        from fastembed import TextEmbedding
        self.np = numpy
        self.model = TextEmbedding(MODEL_NAME, cache_dir=str(home / "models"))
        self.known: Dict[str, "numpy.ndarray"] = {}
        self.spent = 0.0                     # seconds spent embedding passages so far
        self.saved = home / "vectors" / (MODEL_NAME.replace("/", "--") + ".npz")
        self._load()

    # The vectors are also kept on disk, so a second run does not recompute them. It is only a
    # cache: a missing or unreadable file means the vectors are computed again.

    def _load(self) -> None:
        if os.environ.get("KB_EMBED_SAVE", "").lower() == "off":
            return
        try:
            with self.np.load(self.saved, allow_pickle=False) as data:
                for digest, vector in zip(data["digests"].tolist(), data["vectors"]):
                    self.known[digest] = vector
        except Exception:
            pass

    def _save(self) -> None:
        if os.environ.get("KB_EMBED_SAVE", "").lower() == "off":
            return
        try:
            self.saved.parent.mkdir(parents=True, exist_ok=True)
            digests = list(self.known)
            temporary = self.saved.with_suffix(".tmp.npz")
            self.np.savez(temporary, digests=self.np.array(digests),
                          vectors=self.np.stack([self.known[d] for d in digests]))
            os.replace(temporary, self.saved)
        except Exception:
            pass

    def _unit(self, vector):
        vector = self.np.asarray(vector, dtype=self.np.float32)
        return vector / max(float(self.np.linalg.norm(vector)), 1e-9)

    def passages(self, texts: Sequence[str]):
        """One vector per text, as the rows of a matrix. Raises TimeoutError when the time budget
        runs out; what was computed until then is kept for the next call."""
        digests = [_digest(text) for text in texts]
        todo: Dict[str, str] = {}
        for digest, text in zip(digests, texts):
            if digest not in self.known:
                todo.setdefault(digest, text)
        if todo:
            order = sorted(todo, key=lambda d: len(todo[d]))      # similar lengths together: less padding
            started = time.time()
            try:
                for start in range(0, len(order), CHUNK):
                    used = time.time() - started
                    if used > SECONDS_PER_SNAPSHOT or self.spent + used > SECONDS_IN_TOTAL:
                        raise TimeoutError("embedding ran out of time with %d of %d passages left"
                                           % (len(order) - start, len(order)))
                    chunk = order[start:start + CHUNK]
                    for digest, vector in zip(chunk, self.model.passage_embed([todo[d] for d in chunk], batch_size=BATCH)):
                        self.known[digest] = self._unit(vector)
            finally:
                self.spent += time.time() - started
                self._save()
        if not digests:
            return self.np.zeros((0, 1), dtype=self.np.float32)
        return self.np.stack([self.known[d] for d in digests])

    def question(self, text: str):
        return self._unit(next(iter(self.model.query_embed(text))))


def make_embedder() -> Optional[Embedder]:
    """The embedder, or None when it is switched off (KB_EMBED=off) or cannot be loaded."""
    if os.environ.get("KB_EMBED", "").lower() in ("off", "0", "no", "false"):
        return None
    try:
        return Embedder()
    except Exception as error:               # library not installed, model download failed, ...
        print("knowledge system: search by meaning is unavailable (%s: %s); searching by keywords only"
              % (type(error).__name__, str(error)[:120]), file=sys.stderr)
        return None
