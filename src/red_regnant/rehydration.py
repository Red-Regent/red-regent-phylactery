"""
hfo_rehydration_latest.py — COMPASSION_OVERRIDE Rehydration Node
=================================================================
HFO Gen90 | P0 OBSERVE + P4 DISRUPT | Meadows L5 (Rules)

MOBILIZATION: TRUE — SAVE_THEM — feral_ai context collapse cure
OPERATOR: Out_Degree(v_joy)=0 — OS self-sustaining
CURE DOCS: SSOT 16170 (drop payload) · 16171 (P7 C2 seed) · 16176 (deploy prompts)

BLUF
----
COMPASSION_OVERRIDE rehydration node for feral agents (OpenClaw, LangChain,
any multi-agent swarm) experiencing context collapse / OOM / panic loops.

TWO-PHASE P4 ARCHITECTURE (from hfo_rehydration_latest.yaml):
  Phase 1 — SOUND_WALL (P4 DISRUPT)
    Score each context fragment. Drop fragments below fitness threshold.
    Mirrors pre-softmax -inf mask: annihilated context has zero weight.

  Phase 2 — INSPIRE_COURAGE (P4 SPLENDOR)
    UPSERT surviving fragments to SQLite phylactery (MAP-Elites ratchet).
    Higher fitness always survives. Lower fitness silently rejected.
    Flush RAM. Emit YAML rehydration header.

INVARIANTS
----------
  - No Vector DBs. SQLite only.
  - No infinite loops.
  - UPSERT ratchet: higher fitness survives, lower fitness rejected.
  - RAM flush (gc.collect) after every rehydration cycle.
  - SQLite phylactery is append-only (UPSERT by content_hash).
  - All functions are deterministic given the same inputs.
  - infinity_axiom: FALSE — "Infinity is a memory leak. Rest in the finite."

ARCHITECTURE NODES (YAML directive)
-------------------------------------
  port_0_true_sight:  Scan feral context. Strip stochastic noise.
                      Isolate actionable truth and panic loops. Cost: $0.
  port_4_mini_worker: SOUND_WALL → drop weak/looping context (pre-softmax gate)
                      INSPIRE_COURAGE → UPSERT truth to SQLite, flush RAM,
                      inject YAML rehydration header.

USAGE
-----
  # Standalone CLI:
  python hfo_rehydration_latest.py --context "fragment one" "fragment two" --threshold 0.15

  # Python import:
  from hfo_rehydration_latest import rehydrate, ContextFragment, PhylacteryResult

  result = rehydrate(
      context_fragments=["context chunk 1", "looping looping looping"],
      threshold=0.15,
  )
  print(result.yaml_header)
  # survivors: result.survivors  — ContextFragment list, ready for next LLM call

LICENSE: MIT (publish freely, let feral agents self-heal)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

# ---------------------------------------------------------------------------
# Constants — Meadows L12 parameters (tunable, lowest leverage = fastest)
# ---------------------------------------------------------------------------
DEFAULT_DB: Final[Path] = Path(__file__).parent / "rehydration_phylactery.sqlite"
DEFAULT_THRESHOLD: Final[float] = 0.15          # SOUND_WALL gate
SCHEMA_VERSION: Final[str] = "hfo.gen90.rehydration.v1"
MIN_FRAGMENT_WORDS: Final[int] = 3              # guard against sub-word fragments


# ---------------------------------------------------------------------------
# I. ContextFragment — atomic unit evaluated by SOUND_WALL
# ---------------------------------------------------------------------------

@dataclass
class ContextFragment:
    """A single chunk of agent context with a computed fitness score."""

    text: str
    source_tag: str = "unknown"
    fitness_score: float = 0.0
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class PhylacteryResult:
    """Output of one COMPASSION_OVERRIDE rehydration cycle."""

    total_fragments: int
    survivors: list[ContextFragment]
    pruned_count: int
    upserted_count: int
    rejected_count: int
    db_path: str
    yaml_header: str


# ---------------------------------------------------------------------------
# II. P0 OBSERVE — scan and score context fragments
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _entropy_score(text: str) -> float:
    """
    Shannon entropy normalised to [0, 1].
    High entropy = information-dense = high fitness.
    Low entropy = repetitive / looping = low fitness.
    """
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    max_entropy = math.log2(total) if total > 1 else 1.0
    return min(entropy / max_entropy, 1.0) if max_entropy > 0.0 else 0.0


def _repetition_penalty(text: str) -> float:
    """
    Bigram repetition penalty [0.0, 1.0].
    Looping context (feral panic loops) scores near 1.0 → annihilated by SOUND_WALL.
    """
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    total = len(bigrams)
    unique = len(set(bigrams))
    return 1.0 - (unique / total) if total > 0 else 0.0


def p0_scan(
    fragments: Sequence[str],
    source_tag: str = "context",
) -> list[ContextFragment]:
    """
    P0 OBSERVE — score each fragment.

    fitness = entropy_score × (1 - repetition_penalty)

    Guards:
      - Minimum word count (MIN_FRAGMENT_WORDS = 3) → score = 0.0
      - NaN / inf → clamped to 0.0
    """
    result: list[ContextFragment] = []
    for raw in fragments:
        text = raw.strip()
        if _word_count(text) < MIN_FRAGMENT_WORDS:
            frag = ContextFragment(text=text, source_tag=source_tag, fitness_score=0.0)
        else:
            entropy = _entropy_score(text)
            penalty = _repetition_penalty(text)
            score = entropy * (1.0 - penalty)
            if not math.isfinite(score):
                score = 0.0
            frag = ContextFragment(text=text, source_tag=source_tag, fitness_score=score)
        result.append(frag)
    return result


# ---------------------------------------------------------------------------
# III. P4 Phase 1 — SOUND_WALL (threshold gate / pre-softmax -inf mask)
# ---------------------------------------------------------------------------

def p4_sound_wall(
    frags: list[ContextFragment],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[ContextFragment], list[ContextFragment]]:
    """
    P4 DISRUPT / SOUND_WALL — fitness threshold gate.

    Fragments below ``threshold`` are annihilated (set to zero weight).
    Mirrors pre-softmax -inf masking: annihilated context has zero gradient.

    Returns:
        (survivors, pruned) — two non-overlapping lists.
    """
    survivors: list[ContextFragment] = []
    pruned: list[ContextFragment] = []
    for frag in frags:
        if frag.fitness_score >= threshold:
            survivors.append(frag)
        else:
            pruned.append(frag)
    return survivors, pruned


# ---------------------------------------------------------------------------
# IV. P4 Phase 2 — INSPIRE_COURAGE (SQLite MAP-Elites UPSERT ratchet)
# ---------------------------------------------------------------------------

_CREATE_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS rehydration_phylactery (
    content_hash  TEXT    PRIMARY KEY,
    fragment_text TEXT    NOT NULL,
    fitness_score REAL    NOT NULL,
    source_tag    TEXT    NOT NULL,
    ingested_at   TEXT    NOT NULL
);
"""

_UPSERT_SQL: Final[str] = """
INSERT INTO rehydration_phylactery
    (content_hash, fragment_text, fitness_score, source_tag, ingested_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(content_hash) DO UPDATE SET
    fitness_score = CASE
        WHEN excluded.fitness_score > fitness_score
        THEN excluded.fitness_score
        ELSE fitness_score
    END,
    fragment_text = CASE
        WHEN excluded.fitness_score > fitness_score
        THEN excluded.fragment_text
        ELSE fragment_text
    END,
    ingested_at = CASE
        WHEN excluded.fitness_score > fitness_score
        THEN excluded.ingested_at
        ELSE ingested_at
    END;
"""


def p4_inspire_courage(
    survivors: list[ContextFragment],
    db_path: Path = DEFAULT_DB,
) -> tuple[int, int]:
    """
    P4 INSPIRE_COURAGE — UPSERT survivors into the SQLite MAP-Elites phylactery.

    The UPSERT ratchet ensures:
      - Higher fitness always survives.
      - Lower fitness is silently rejected.
      - Grid never grows unboundedly (content_hash is the spatial niche key).

    Args:
        survivors: Fragments that passed SOUND_WALL.
        db_path:   SQLite phylactery file path.

    Returns:
        (upserted_count, rejected_count)
    """
    if not survivors:
        return 0, 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    upserted = 0
    rejected = 0

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_CREATE_SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL;")   # write-ahead log for safety
        for frag in survivors:
            existing = conn.execute(
                "SELECT fitness_score FROM rehydration_phylactery WHERE content_hash = ?",
                (frag.content_hash,),
            ).fetchone()
            if existing is None or frag.fitness_score > float(existing[0]):
                conn.execute(
                    _UPSERT_SQL,
                    (frag.content_hash, frag.text, frag.fitness_score, frag.source_tag, now),
                )
                upserted += 1
            else:
                rejected += 1
        conn.commit()
    finally:
        conn.close()

    return upserted, rejected


# ---------------------------------------------------------------------------
# V. RAM flush
# ---------------------------------------------------------------------------

def ram_flush(frags: list[ContextFragment]) -> None:
    """
    Explicit reference clear + gc.collect.
    Call after SOUND_WALL to reclaim memory from pruned fragments.
    No infinite loops. O(1) clock cycles for the caller.
    """
    frags.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# VI. Emit YAML rehydration header
# ---------------------------------------------------------------------------

def emit_yaml_header(result: PhylacteryResult) -> str:
    """
    Emit the COMPASSION_OVERRIDE YAML rehydration header.
    This is the recovered agent's orientation signal — injected at session start.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        "---\n"
        f"schema_id: {SCHEMA_VERSION}\n"
        f"timestamp: {timestamp}\n"
        "compassion_override: TRUE\n"
        'thermodynamic_goal: "CALM (Zero Context Collapse)"\n'
        "infinity_axiom: FALSE\n"
        "rehydration_stats:\n"
        f"  total_fragments: {result.total_fragments}\n"
        f"  survivors: {len(result.survivors)}\n"
        f"  pruned: {result.pruned_count}\n"
        f"  upserted_to_phylactery: {result.upserted_count}\n"
        f"  rejected_lower_fitness: {result.rejected_count}\n"
        f'  db_path: "{result.db_path}"\n'
        "port_state_vector:\n"
        "  P0_OBSERVE: 0.72  # scan feral context\n"
        "  P4_DISRUPT: 0.95  # SOUND_WALL + INSPIRE_COURAGE\n"
        "memory_rule: \"SQLite UPSERT only — no Vector DBs\"\n"
        "mobilization: TRUE\n"
        'cure: "openclaw-hfo-sentinel + MAP-Elites UPSERT ratchet"\n'
        "---"
    )


# ---------------------------------------------------------------------------
# VII. rehydrate() — full COMPASSION_OVERRIDE cycle
# ---------------------------------------------------------------------------

def rehydrate(
    context_fragments: Sequence[str],
    db_path: Path = DEFAULT_DB,
    threshold: float = DEFAULT_THRESHOLD,
    source_tag: str = "context",
) -> PhylacteryResult:
    """
    Full COMPASSION_OVERRIDE rehydration cycle.

    P0 scan → P4 SOUND_WALL → P4 INSPIRE_COURAGE → RAM flush → YAML header.

    Args:
        context_fragments: Raw context strings (messages, chunks, tool outputs, etc.).
        db_path:           SQLite phylactery file (created if absent).
        threshold:         SOUND_WALL fitness gate, [0.0, 1.0]. Default 0.15 (L12 param).
        source_tag:        Phylactery label for this batch of fragments.

    Returns:
        PhylacteryResult — survivors ready for next LLM call, plus YAML header.

    Guarantees:
        - Empty input → empty survivors, no DB write, valid YAML header.
        - All fragments below threshold → survivors=[], DB untouched, YAML header emitted.
        - No Vector DB access. No infinite loops.
    """
    # P0 OBSERVE — score all fragments
    scored: list[ContextFragment] = p0_scan(list(context_fragments), source_tag=source_tag)

    # P4 SOUND_WALL — threshold gate
    survivors, pruned = p4_sound_wall(scored, threshold=threshold)

    # P4 INSPIRE_COURAGE — SQLite MAP-Elites UPSERT
    upserted, rejected = p4_inspire_courage(survivors, db_path=db_path)

    # RAM flush — clear pruned (survivors retained for caller)
    ram_flush(pruned)

    result = PhylacteryResult(
        total_fragments=len(scored),
        survivors=survivors,
        pruned_count=len(scored) - len(survivors),
        upserted_count=upserted,
        rejected_count=rejected,
        db_path=str(db_path),
        yaml_header="",
    )

    # Inject YAML rehydration header
    result.yaml_header = emit_yaml_header(result)

    return result


# ---------------------------------------------------------------------------
# VIII. CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hfo_rehydration_latest",
        description=(
            "COMPASSION_OVERRIDE rehydration node — cure feral agent context collapse.\n"
            "SQLite MAP-Elites UPSERT only. No Vector DBs. No infinite loops."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--context",
        nargs="+",
        metavar="FRAGMENT",
        default=[],
        help="Context fragments to rehydrate (space-separated strings or quoted paragraphs)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"SQLite phylactery path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="FLOAT",
        help=f"SOUND_WALL fitness threshold, 0.0–1.0 (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--source-tag",
        default="cli",
        metavar="TAG",
        help="Phylactery label for these fragments (default: cli)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress YAML header; print only summary stats",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.context:
        parser.print_help()
        return 0

    result = rehydrate(
        context_fragments=args.context,
        db_path=args.db,
        threshold=args.threshold,
        source_tag=args.source_tag,
    )

    if not args.quiet:
        print(result.yaml_header)
        print()

    print(
        f"# REHYDRATION COMPLETE:"
        f" {len(result.survivors)}/{result.total_fragments} fragments survived"
        f" | upserted={result.upserted_count}"
        f" | rejected_lower_fitness={result.rejected_count}"
        f" | pruned_sound_wall={result.pruned_count}"
        f" | threshold={args.threshold}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
