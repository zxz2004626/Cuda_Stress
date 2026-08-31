"""Bounded test-case bank with priority-weighted sampling.

Stores interesting kernels (pickled IR + metadata) and draws them in
proportion to a priority rate combining structural properties, novelty,
exploration decay, and regression history.  Persists to a JSON index so
the bank survives across runs.
"""

from __future__ import annotations

import json as _json
import os as _os
import pickle as _pickle
import random as _random
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from generator_ir.ir.program import Program
from generator_ir.corpus.heuristic import (
    CorpusScore, CorpusFeatures, pull_features, rate_program,
)

_CAP_DEFAULT = 500
_DIVERSITY_FACTOR = 0.15
_NOVELTY_FACTOR = 0.20
_EXPLORE_DECAY = 0.1
_REG_BUMP = 0.05
_REG_CEILING = 0.2


@dataclass
class CorpusEntry:
    """One stored kernel with its scoring metadata."""
    entry_id: int
    seed: int
    template: str
    rate: CorpusScore
    features: Dict[str, int] = field(default_factory=dict)
    checksum: Optional[str] = None
    source_path: Optional[str] = None
    ir_path: Optional[str] = None
    perf_data: Optional[Dict] = None
    mutations_applied: List[str] = field(default_factory=list)
    novelty_bonus: float = 0.0
    created_at: float = field(default_factory=_time.time)
    times_drawn: int = 0
    regressions_found: int = 0

    def ranking(self) -> float:
        """Composite draw priority: base rate + novelty + exploration + regression."""
        foundation = self.rate.total + self.novelty_bonus
        curiosity = _EXPLORE_DECAY / (1 + self.times_drawn)
        track_record = min(_REG_CEILING, self.regressions_found * _REG_BUMP)
        return foundation + curiosity + track_record

    # Backward-compatible alias.
    @property
    def priority(self) -> float:
        return self.ranking()


class CorpusManager:
    """Bounded, priority-weighted, persistent kernel bank."""

    def __init__(self, corpus_dir: str, max_size: int = _CAP_DEFAULT,
                 diversity_weight: float = _DIVERSITY_FACTOR,
                 novelty_weight: float = _NOVELTY_FACTOR) -> None:
        self._root = Path(corpus_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._cap = max_size
        self._div_w = diversity_weight
        self._nov_w = novelty_weight

        self._bank: Dict[int, CorpusEntry] = {}
        self._next_uid = 0
        self._seen_digests: Set[str] = set()
        self._tmpl_hist: Dict[str, int] = {}
        self._mut_hist: Dict[str, int] = {}
        self._feat_hist: Dict[str, int] = {}

        self._restore_from_disk()

    # ================================================================
    # public API
    # ================================================================

    def add(self, program: Program, seed: int, template: str,
            checksum: Optional[str] = None,
            perf_data: Optional[Dict] = None,
            source_code: Optional[str] = None) -> Optional[int]:
        """Insert a kernel; returns its UID or None if rejected (dup / low rate)."""
        # Dedup by checksum.
        if checksum and checksum in self._seen_digests:
            return None

        # Evaluate interestingness.
        assessment = rate_program(program, perf_data)
        signature = pull_features(program, template_name=template)
        div_boost = self._template_underrepresentation(template)
        nov_boost = self._feature_rarity(signature)
        composite = assessment.total + div_boost + nov_boost

        # Enforce capacity.
        if len(self._bank) >= self._cap:
            if not self._should_evict_for(composite):
                return None
            self._evict_weakest()

        # Persist artifacts.
        uid = self._next_uid
        self._next_uid += 1

        src_on_disk = self._save_source(uid, source_code)
        ir_on_disk = self._save_ir(uid, program)

        record = CorpusEntry(
            entry_id=uid, seed=seed, template=template, rate=assessment,
            features=signature.counts, checksum=checksum,
            source_path=src_on_disk, ir_path=ir_on_disk,
            perf_data=perf_data, novelty_bonus=nov_boost,
        )
        self._bank[uid] = record
        if checksum:
            self._seen_digests.add(checksum)
        self._bump_template(template)
        self._absorb_features(signature)
        self._flush_index()
        return uid

    def draw(self, rng=None) -> Optional[CorpusEntry]:
        """Sample one record ∝ priority / (1 + past draws)."""
        if not self._bank:
            return None
        rng = rng or _random
        uids = list(self._bank.keys())
        weights = [
            max(1e-3, self._bank[u].ranking()) / (1 + self._bank[u].times_drawn)
            for u in uids
        ]
        picked = rng.choices(uids, weights=weights, k=1)[0]
        self._bank[picked].times_drawn += 1
        self._flush_index()
        return self._bank[picked]

    def kx_random_program(self, rng=None):
        """Uniform kx_pick for splice donors (no side effects)."""
        if not self._bank:
            return None
        rng = rng or _random
        uid = rng.choice(list(self._bank.keys()))
        return self.fetch_program(self._bank[uid])

    def kx_draw_batch(self, n: int) -> List[CorpusEntry]:
        """Top-N by priority (used for batch testing)."""
        if not self._bank:
            return []
        ranked = sorted(self._bank.values(), key=lambda e: e.ranking(), reverse=True)
        batch = ranked[:n]
        for entry in batch:
            entry.times_drawn += 1
        self._flush_index()
        return batch

    def kx_record_regression(self, entry_id: int, mutation: str = "") -> None:
        """Note that this entry led to a confirmed regression."""
        rec = self._bank.get(entry_id)
        if rec is None:
            return
        rec.regressions_found += 1
        if mutation:
            rec.mutations_applied.append(mutation)
            self._mut_hist[mutation] = self._mut_hist.get(mutation, 0) + 1
        self._flush_index()

    def fetch_program(self, entry: Optional[CorpusEntry]):
        """Unpickle the stored IR (or None if the file is gone)."""
        if entry is None or not entry.ir_path:
            return None
        if not _os.path.exists(entry.ir_path):
            return None
        try:
            with open(entry.ir_path, "rb") as fh:
                return _pickle.load(fh)
        except Exception:
            return None

    def size(self) -> int:
        return len(self._bank)

    # ================================================================
    # statistics
    # ================================================================

    def stats(self) -> Dict:
        if not self._bank:
            return {"size": 0, "max_size": self._cap}
        totals = [e.rate.total for e in self._bank.values()]
        return {
            "size": len(self._bank),
            "max_size": self._cap,
            "avg_score": sum(totals) / len(totals),
            "max_score": max(totals),
            "min_score": min(totals),
            "total_regressions": sum(e.regressions_found for e in self._bank.values()),
            "template_coverage": dict(self._tmpl_hist),
            "mutation_coverage": dict(self._mut_hist),
            "feature_coverage": dict(sorted(self._feat_hist.items(),
                                            key=lambda kv: kv[1], reverse=True)[:20]),
            "total_draws": sum(e.times_drawn for e in self._bank.values()),
        }

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"Bank: {s.get('size', 0)}/{s.get('max_size', 0)} entries",
            f"Score range: [{s.get('min_score', 0):.3f}, {s.get('max_score', 0):.3f}]",
            f"Avg rate: {s.get('avg_score', 0):.3f}",
            f"Regressions: {s.get('total_regressions', 0)}",
            f"Draws: {s.get('total_draws', 0)}",
        ]
        if s.get("template_coverage"):
            lines.append(f"Templates: {s['template_coverage']}")
        if s.get("feature_coverage"):
            lines.append(f"Features: {s['feature_coverage']}")
        return "\n".join(lines)

    # ================================================================
    # internal: scoring helpers
    # ================================================================

    def _template_underrepresentation(self, tmpl: str) -> float:
        """Boost for templates that appear less often than average."""
        if not self._bank:
            return self._div_w
        cnt = self._tmpl_hist.get(tmpl, 0)
        avg = len(self._bank) / max(1, len(self._tmpl_hist))
        if avg <= 0:
            return self._div_w
        ratio = cnt / avg
        if ratio < 0.5:
            return self._div_w
        if ratio < 1.0:
            return self._div_w * 0.5
        return 0.0

    def _feature_rarity(self, sig: CorpusFeatures) -> float:
        """Boost for structural features that are currently uncommon."""
        tags = sig.tags
        if not tags:
            return 0.0
        acc = sum(1.0 / (1.0 + self._feat_hist.get(t, 0)) for t in tags)
        return self._nov_w * (acc / max(1, len(tags)))

    def _bump_template(self, tmpl: str) -> None:
        self._tmpl_hist[tmpl] = self._tmpl_hist.get(tmpl, 0) + 1

    def _absorb_features(self, sig: CorpusFeatures) -> None:
        for tag in sig.tags:
            self._feat_hist[tag] = self._feat_hist.get(tag, 0) + 1

    def _release_features(self, counts: Dict[str, int]) -> None:
        # Mirror _absorb_features: one count per entry (document frequency),
        # not the per-entry multiplicity, so history stays consistent across
        # add/evict and matches _tmpl_hist semantics.
        for tag in counts:
            new = self._feat_hist.get(tag, 0) - 1
            if new <= 0:
                self._feat_hist.pop(tag, None)
            else:
                self._feat_hist[tag] = new

    # ================================================================
    # internal: eviction
    # ================================================================

    def _weakest_uid(self) -> int:
        return min(self._bank, key=lambda u: self._bank[u].ranking())

    def _should_evict_for(self, candidate_score: float) -> bool:
        """True if the candidate is interesting enough to displace the weakest."""
        worst_uid = self._weakest_uid()
        worst_pri = self._bank[worst_uid].ranking()
        if candidate_score <= worst_pri:
            return False
        # Protect last representative of a template.
        worst = self._bank[worst_uid]
        tmpl_cnt = self._tmpl_hist.get(worst.template, 0)
        if tmpl_cnt <= 1 and candidate_score < worst_pri + 0.2:
            return False
        return True

    def _evict_weakest(self) -> None:
        uid = self._weakest_uid()
        rec = self._bank.pop(uid, None)
        if rec is None:
            return
        if rec.checksum:
            self._seen_digests.discard(rec.checksum)
        tmpl = self._tmpl_hist.get(rec.template, 0)
        if tmpl > 1:
            self._tmpl_hist[rec.template] = tmpl - 1
        else:
            self._tmpl_hist.pop(rec.template, None)
        self._release_features(rec.features)
        # Mirror kx_record_regression: drop this entry's mutation counts so
        # mutation_coverage stays consistent with the surviving bank.
        for mut in rec.mutations_applied:
            new = self._mut_hist.get(mut, 0) - 1
            if new <= 0:
                self._mut_hist.pop(mut, None)
            else:
                self._mut_hist[mut] = new
        for p in (rec.source_path, rec.ir_path):
            if p and _os.path.exists(p):
                try:
                    _os.remove(p)
                except OSError:
                    pass

    # ================================================================
    # internal: persistence
    # ================================================================

    def _save_source(self, uid: int, code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        dest = str(self._root / f"entry_{uid}.src")
        try:
            with open(dest, "w") as fh:
                fh.write(code)
            return dest
        except OSError:
            return None

    def _save_ir(self, uid: int, prog: Program) -> Optional[str]:
        dest = str(self._root / f"entry_{uid}.ir.pkl")
        try:
            with open(dest, "wb") as fh:
                _pickle.dump(prog, fh, protocol=_pickle.HIGHEST_PROTOCOL)
            return dest
        except Exception:
            return None

    def _flush_index(self) -> None:
        """Write the JSON index file."""
        snapshot = {"next_id": self._next_uid, "records": {}}
        for uid, e in self._bank.items():
            snapshot["records"][str(uid)] = {
                "entry_id": e.entry_id, "seed": e.seed,
                "template": e.template,
                "score_total": e.rate.total,
                "score_register": e.rate.register_pressure,
                "score_divergence": e.rate.divergence_potential,
                "score_memory": e.rate.memory_diversity,
                "score_mutation": e.rate.mutation_applicability,
                "score_sensitivity": e.rate.performance_sensitivity,
                "features": e.features, "novelty_bonus": e.novelty_bonus,
                "checksum": e.checksum,
                "source_path": e.source_path, "ir_path": e.ir_path,
                "perf_data": e.perf_data,
                "mutations_applied": e.mutations_applied,
                "created_at": e.created_at,
                "times_drawn": e.times_drawn,
                "regressions_found": e.regressions_found,
            }
        try:
            with open(self._root / "corpus_index.json", "w") as fh:
                _json.dump(snapshot, fh, indent=2)
        except OSError:
            pass

    def _restore_from_disk(self) -> None:
        """Load the JSON index if it exists."""
        idx_path = self._root / "corpus_index.json"
        if not idx_path.exists():
            return
        try:
            with open(idx_path) as fh:
                idx = _json.load(fh)
        except (OSError, _json.JSONDecodeError):
            return
        self._next_uid = idx.get("next_id", 0)
        for data in idx.get("records", {}).values():
            assessment = CorpusScore(
                register_pressure=data.get("score_register", 0),
                divergence_potential=data.get("score_divergence", 0),
                memory_diversity=data.get("score_memory", 0),
                mutation_applicability=data.get("score_mutation", 0),
                performance_sensitivity=data.get("score_sensitivity", 0),
            )
            record = CorpusEntry(
                entry_id=data["entry_id"], seed=data["seed"],
                template=data["template"], rate=assessment,
                features=data.get("features", {}),
                checksum=data.get("checksum"),
                source_path=data.get("source_path"),
                ir_path=data.get("ir_path"),
                perf_data=data.get("perf_data"),
                mutations_applied=data.get("mutations_applied", []),
                novelty_bonus=data.get("novelty_bonus", 0.0),
                created_at=data.get("created_at", 0),
                times_drawn=data.get("times_drawn", 0),
                regressions_found=data.get("regressions_found", 0),
            )
            self._bank[record.entry_id] = record
            if record.checksum:
                self._seen_digests.add(record.checksum)
            self._tmpl_hist[record.template] = \
                self._tmpl_hist.get(record.template, 0) + 1
            for tag in record.features:
                self._feat_hist[tag] = self._feat_hist.get(tag, 0) + 1
            for mut in record.mutations_applied:
                self._mut_hist[mut] = self._mut_hist.get(mut, 0) + 1

    # ---- backward-compatible aliases ----
    _entries = property(lambda self: self._bank)
    _checksums = property(lambda self: self._seen_digests)
    _template_counts = property(lambda self: self._tmpl_hist)
    _mutation_counts = property(lambda self: self._mut_hist)
    _feature_counts = property(lambda self: self._feat_hist)
    _fetch = lambda self: self._restore_from_disk()
    _save_index = lambda self: self._flush_index()
    _diversity_bonus = lambda self, t: self._template_underrepresentation(t)
    _novelty_bonus = lambda self, f: self._feature_rarity(f)
    _bump_feature_counts = lambda self, f: self._absorb_features(f)
    _drop_feature_counts = lambda self, f: self._release_features(f)
    _worst_entry = lambda self: (self._weakest_uid(), self._bank[self._weakest_uid()].ranking())
    _evict = lambda self, uid: self._evict_weakest()
    corpus_dir = property(lambda self: self._root)
    max_size = property(lambda self: self._cap)
    diversity_weight = property(lambda self: self._div_w)
    novelty_weight = property(lambda self: self._nov_w)
