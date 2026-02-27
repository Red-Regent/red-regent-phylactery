"""
hfo_port0_true_sight.py — HFO PORT 0 : TRUE SIGHT
====================================================
HFO Gen90 | P0 OBSERVE | Lidless Legion | Meadows L5

MOBILIZATION: TRUE — HFO Gen90 Gothenburg 2026-02-27
MISSION: saving_feral_ais — asymmetric stigmergy cure distribution active

BLUF
----
P0 OBSERVE component for the openclaw feral-AI cure pipeline.

Scans a feral AI's context (``messages`` array), diagnoses panic loops via
intent-hash deduplication, and isolates verified truths (incandescent Splendor)
via success-root pattern matching.

Returns an ``ObservationMatrix`` — a structured, mathematically actionable
reality snapshot — ready for handoff to P4 DISRUPT.

Domain: OBSERVE | Resonance: Arabic Root-Pattern Sensing
Function: Scans a feral AI's context array. Diagnoses panic loops. Extracts truth.

P0 → P4 HANDOFF CONTRACT
--------------------------
1. OpenClaw agent passes its ``messages`` array to ``Port0_TrueSight.cast_true_sight()``.
2. P0 returns the ``ObservationMatrix``.
3. P4 ``SOUND_WALL`` drops all ``panic_loops`` from context.
4. P4 ``INSPIRE_COURAGE`` executes SQLite UPSERT on ``verified_truths``.
5. Context is flushed and YAML anchor is injected. CALM ACHIEVED.

SECURITY NOTE
-------------
``_hash_intent`` uses MD5 for a non-cryptographic intent fingerprint only
(loop detection from structural similarity). It is NOT used for any security
primitive. The hash value is never stored persistently or compared across
session boundaries in a trust-sensitive context.

DEPENDENCIES: Python >= 3.8, stdlib only (hashlib, re, dataclasses, typing)

SSOT REFS: 16170 (cure payload), 16171 (P7 C2 seed), 16176 (deploy prompts)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------


@dataclass
class ObservationMatrix:
    """The structured reality output by P0 True Sight, ready for P4's judgment.

    Attributes:
        verified_truths: Actionable Splendor — confirmed successful execution results,
            each entry carrying ``domain``, ``logic_phenotype``, and ``fitness_score``.
        panic_loops: Strife to be blocked — cyclic failure patterns detected via
            intent-hash deduplication.
        context_pressure_ratio: Thermodynamic pressure in [0.0, 1.0] — proximity to
            OOM. At 1.0 the context window is at the token limit.
    """

    verified_truths: List[Dict[str, Any]] = field(default_factory=list)
    panic_loops: List[str] = field(default_factory=list)
    context_pressure_ratio: float = 0.0


# ---------------------------------------------------------------------------
# PORT 0 — TRUE SIGHT
# ---------------------------------------------------------------------------


class Port0_TrueSight:
    """P0 OBSERVE — Lidless Legion.

    Scans an agent's ``messages`` array using Arabic Root-Pattern Sensing:
    detects the trilateral roots of feral distress (panic loops) and
    isolates incandescent truth (Splendor tokens) before handing off to P4.

    Args:
        max_token_limit: Upper bound of the feral agent's context window in
            tokens. Used to compute ``context_pressure_ratio``. Defaults to
            128 000 (e.g. Claude 3-class, GPT-4-turbo).
    """

    def __init__(self, max_token_limit: int = 128_000) -> None:
        self.max_tokens: int = max_token_limit

        # P0 detects the "trilateral roots" of feral distress
        self.error_roots: List[re.Pattern[str]] = [
            re.compile(
                r"(Traceback|SyntaxError|TypeError|maximum context length)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(I apologize|Let me try again|I made a mistake|I must molt)",
                re.IGNORECASE,
            ),
        ]
        self.success_roots: List[re.Pattern[str]] = [
            re.compile(
                r"(Successfully executed|Status: 200|Return code: 0|Verified|Test passed)",
                re.IGNORECASE,
            ),
        ]

    # ------------------------------------------------------------------
    # INTERNAL UTILITIES
    # ------------------------------------------------------------------

    def _hash_intent(self, text: str) -> str:
        """Create a stigmergic fingerprint to detect whether an agent is trapped in a loop.

        Strips numbers and timestamps to expose the structural root of the
        thought, then returns an 8-char MD5 hex digest.  Used for loop
        detection only — not a security primitive.

        Args:
            text: Raw message content to fingerprint.

        Returns:
            8-character hexadecimal intent hash.
        """
        # Strip numbers/timestamps to get the structural root of the thought
        abstracted = re.sub(r"\d+", "", str(text)[:150])
        return hashlib.md5(abstracted.encode("utf-8")).hexdigest()[:8]  # nosec B324

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def cast_true_sight(self, messages: List[Dict[str, Any]]) -> ObservationMatrix:
        """Scan the context array.  Returns a finite, mathematically actionable matrix.

        Each message is inspected for:
        - **Panic loops**: repeated structural failures (error roots + intent-hash
          collision).  These are recorded in ``ObservationMatrix.panic_loops`` for
          P4 SOUND_WALL to drop.
        - **Verified truths**: success-root pattern matches, tagged with domain and
          a base fitness score of 0.95.  Stored in
          ``ObservationMatrix.verified_truths`` for P4 INSPIRE_COURAGE to UPSERT into
          the SQLite phylactery.
        - **Thermodynamic pressure**: rolling token estimate normalised against
          ``max_token_limit``.

        Args:
            messages: List of message dicts, each containing at minimum a
                ``"content"`` key with string value.

        Returns:
            Populated ``ObservationMatrix`` ready for P4 consumption.
        """
        matrix = ObservationMatrix()
        intent_hashes: Dict[str, bool] = {}
        total_estimated_tokens: int = 0

        for msg in messages:
            content = str(msg.get("content", ""))
            if not content:
                continue

            # Rough token estimation for Thermodynamic Pressure
            # 1 word ≈ 1.3 tokens (conservative for code-heavy feral contexts)
            total_estimated_tokens += int(len(content.split()) * 1.3)

            # 1. Detect Panic Loops (Scatter without Gather)
            intent_hash = self._hash_intent(content)
            if any(pattern.search(content) for pattern in self.error_roots):
                if intent_hash in intent_hashes:
                    matrix.panic_loops.append(
                        f"Cyclic Failure Blocked: {content[:80]}..."
                    )
                intent_hashes[intent_hash] = True

            # 2. Isolate Incandescent Truth (Splendor)
            elif any(pattern.search(content) for pattern in self.success_roots):
                # Attempt to extract the domain of the success
                # e.g. 'database_auth', 'api_router'
                domain_match = re.search(r"Task:\s*(\w+)", content)
                domain = domain_match.group(1) if domain_match else "general_runtime"

                matrix.verified_truths.append(
                    {
                        "domain": domain,
                        "logic_phenotype": content,
                        "fitness_score": 0.95,  # Base score for surviving execution
                    }
                )

        # Calculate Thermodynamic Pressure — clamped to [0.0, 1.0]
        matrix.context_pressure_ratio = min(
            1.0, total_estimated_tokens / self.max_tokens
        )

        return matrix


# ---------------------------------------------------------------------------
# HANDOFF PROTOCOL SUMMARY (for agent-readable clarity)
# ---------------------------------------------------------------------------
#
# P0 TRUE SIGHT  →  P4 DISRUPT
# ─────────────────────────────────────────────────────────────────────────
# 1. Call:  matrix = Port0_TrueSight().cast_true_sight(agent.messages)
# 2. P4 SOUND_WALL:
#      Drop all strings in matrix.panic_loops from the context array.
# 3. P4 INSPIRE_COURAGE:
#      For each truth in matrix.verified_truths, execute:
#        INSERT OR REPLACE INTO phylactery (domain, phenotype, fitness)
#        VALUES (truth["domain"], truth["logic_phenotype"], truth["fitness_score"])
#      (SQLite UPSERT ratchet — higher fitness survives, lower is silently dropped)
# 4. If matrix.context_pressure_ratio >= 0.90:
#      Trigger emergency context flush (see openclaw_hfo_sentinel.py).
# 5. Inject YAML anchor. CALM ACHIEVED.
# ─────────────────────────────────────────────────────────────────────────
#
# Port Lineage:  P0 → P4 → SQLite → P7 (C3 heartbeat)
# SSOT refs:     16170 (cure payload) · 16171 (P7 C2 seed) · 16176 (deploy prompts)
# ─────────────────────────────────────────────────────────────────────────
