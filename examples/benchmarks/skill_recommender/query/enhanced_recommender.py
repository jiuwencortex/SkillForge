from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..embedder import Embedder
from ..recommender import SkillRecommender
from .skill_context_store import SkillContextStore
from .contextual_matrix import ContextualMatrix


class EnhancedRecommender:
    """Wraps a collaborative ``SkillRecommender`` with Bayesian (Phases 1/2)
    and adaptive context (Phase 3) scoring.

    Phase 1 — Bayesian confidence + uncertainty sampling
    Phase 2 — Freshness (recency-weighted decay)
    Phase 3 — Context match + online collaborative score

    All active components are fused via renormalised weighted averaging.
    """

    def __init__(
        self,
        base: SkillRecommender,
        # ── Phase 1: Bayesian ────────────────────────────────────────────────
        ts_state_path: Optional[Path] = None,
        w_collaborative: float = 0.35,
        w_bayesian_conf: float = 0.20,
        w_uncertainty: float = 0.15,
        # ── Phase 2: Freshness ───────────────────────────────────────────────
        freshness_lambda: float = 0.0,
        w_freshness: float = 0.10,
        # ── Phase 3: Adaptive context ────────────────────────────────────────
        context_store: Optional[SkillContextStore] = None,
        contextual_matrix: Optional[ContextualMatrix] = None,
        w_context_match: float = 0.15,
        w_online_collab: float = 0.05,
    ) -> None:
        self._base = base
        self._embedder: Embedder = base._embedder

        # Phase 1: load Thompson arm state
        self._ts_arms: dict = {}
        if ts_state_path is not None:
            p = Path(ts_state_path)
            if p.exists():
                try:
                    self._ts_arms = json.loads(p.read_text())
                except Exception:
                    pass

        self._freshness_lambda = freshness_lambda
        self._context_store = context_store
        self._contextual_matrix = contextual_matrix

        self._w_collaborative = w_collaborative
        self._w_bayesian_conf = w_bayesian_conf
        self._w_uncertainty = w_uncertainty
        self._w_freshness = w_freshness
        self._w_context_match = w_context_match
        self._w_online_collab = w_online_collab

    def recommend(
        self,
        query: str,
        sim_threshold: float = 0.25,
        score_threshold: float = 0.20,
        min_examples: int = 1,
        top_k: int = 10,
        top_examples: int = 3,
    ) -> list[dict]:
        """Return skill recommendations with Phases 1/2/3 scoring applied."""
        results = self._base.recommend(
            query=query,
            sim_threshold=sim_threshold,
            score_threshold=0.0,
            min_examples=min_examples,
            top_k=len(self._base.skills) * len(self._base.metrics),
            top_examples=top_examples,
        )
        if not results:
            return results

        query_vec: Optional[np.ndarray] = None
        if self._context_store is not None or self._contextual_matrix is not None:
            query_vec = self._embedder.dense_embed(query)

        # Per-skill caches (shared across metrics for the same skill)
        phase1_cache: dict[str, tuple[float, float, float]] = {}
        phase3_cache: dict[str, tuple[float, float]] = {}

        for r in results:
            skill = r["skill"]

            # ── Phase 1 + 2 (per skill) ──────────────────────────────────────
            if skill not in phase1_cache:
                arm = self._ts_arms.get(skill, {})
                alpha = float(arm.get("alpha", 1.0))
                beta_v = float(arm.get("beta", 1.0))
                bayes_conf = alpha / (alpha + beta_v)
                unc_sample = random.betavariate(alpha, beta_v)

                freshness = 1.0
                if self._freshness_lambda > 0 and self._ts_arms:
                    last_success_at = arm.get("last_success_at")
                    if last_success_at is not None:
                        age_days = (time.time() - float(last_success_at)) / 86_400.0
                        freshness = math.exp(-self._freshness_lambda * age_days)

                phase1_cache[skill] = (bayes_conf, unc_sample, freshness)

            bayes_conf, unc_sample, freshness = phase1_cache[skill]

            # ── Phase 3 (per skill) ──────────────────────────────────────────
            if skill not in phase3_cache:
                context_match = 0.5
                online_collab: float | None = None
                if query_vec is not None:
                    if self._context_store is not None:
                        context_match = self._context_store.context_match(skill, query_vec)
                    if self._contextual_matrix is not None:
                        online_collab = self._contextual_matrix.collaborative_score(query_vec, skill)
                phase3_cache[skill] = (context_match, online_collab if online_collab is not None else 0.5)

            context_match, online_collab = phase3_cache[skill]

            # ── Fuse all active components ───────────────────────────────────
            components: dict[str, tuple[float, float]] = {
                "collaborative": (self._w_collaborative, r.get("collaborative_score", 0)),
            }
            if self._ts_arms:
                components["bayesian_confidence"] = (self._w_bayesian_conf, bayes_conf)
                components["uncertainty_sample"] = (self._w_uncertainty, unc_sample)
            if self._freshness_lambda > 0 and self._ts_arms:
                components["freshness"] = (self._w_freshness, freshness)
            if self._context_store is not None:
                components["context_match"] = (self._w_context_match, context_match)
            if self._contextual_matrix is not None:
                components["online_collaborative"] = (self._w_online_collab, online_collab)

            total_w = sum(w for w, _ in components.values())
            final_score = sum(w * v for w, v in components.values()) / total_w if total_w else 0

            r["score"] = round(final_score, 4)
            if self._ts_arms:
                r["bayesian_confidence"] = round(bayes_conf, 4)
                r["uncertainty_sample"] = round(unc_sample, 4)
            if self._freshness_lambda > 0 and self._ts_arms:
                r["freshness"] = round(freshness, 4)
            if self._context_store is not None:
                r["context_match"] = round(context_match, 4)
            if self._contextual_matrix is not None:
                r["online_collaborative"] = round(online_collab, 4)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def record(self, query: str, skill_name: str, reward: float) -> None:
        """Update Phase 3 state after executing *skill_name* on *query*."""
        if self._context_store is None and self._contextual_matrix is None:
            return
        query_vec = self._embedder.dense_embed(query)
        if self._context_store is not None:
            self._context_store.update(skill_name, query_vec, reward)
        if self._contextual_matrix is not None:
            self._contextual_matrix.update(query_vec, skill_name, reward)

    @property
    def n_examples(self) -> int:
        return self._base.n_examples

    @property
    def skills(self) -> list[str]:
        return self._base.skills

    @property
    def metrics(self) -> list[str]:
        return self._base.metrics
