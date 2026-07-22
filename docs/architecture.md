# SkillForge Architecture

## Overview

SkillForge implements the **offline track** of Jiuwen's skill-management system.
Given a SKILL.md and a test dataset, it runs a GEPA population-based evolutionary
search to rewrite the skill body, then clears a regression-aware holdout gate
before writing the new version to disk.

The online counterpart — live-session review and lifecycle curation — lives in
**SkillTend**.

---

## 9-Stage Pipeline

`evolve_single_skill()` in `evolvers/skill_evolver_single.py` runs the stages
in sequence:

```
SKILL.md
   │
   ▼
┌─────────────────────────────────────┐
│  Stage 1: skill_finder_and_loader   │  Load SKILL.md + prior metrics
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 2: constraint_validator      │  Size + frontmatter check
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 3: dataset_builder           │  synthetic | golden | external | trajectory
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 4: dspy_configurator         │  DSPy LM init + train/val/holdout split
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 5: gepa_optimizer            │  Population-based GEPA search
│                                     │  (Thompson Level 2: example selector)
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 6: holdout_evaluator         │  Score baseline vs candidate on holdout
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 7: acceptance_gates          │  Threshold gate + Thompson gate
│                                     │  + per-dimension no-regression constraint
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 8: results_display           │  Rich terminal table
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 9: output_saver              │  evolved_skill.md + metrics.json
└─────────────────────────────────────┘
```

---

## Thompson Sampling — 3 Levels

### Level 1: Skill Scheduler (`batch_selection/`)

Used in batch mode (`evolve_skills_batch` / `skillforge --all`).
Maintains one Beta-Bernoulli arm per skill; concentrates budget on skills that
have previously improved.

- Run accepted → α += 1
- Run rejected → β += 1

### Level 2: Training Example Selector (`stage05_gepa_optimizer`)

Selects which training examples drive each GEPA mutation cycle. Examples that
have produced constructive mutations before are preferred.

### Level 3: Bayesian Acceptance Gate (`stage07_acceptance_gates/_thompson_gate.py`)

Requires P(evolved > baseline) ≥ 0.75 (configurable) before deploying.

---

## 7 Fitness Metrics

Alternatives — the operator picks one per run:

| Metric | What it measures | Best for |
|---|---|---|
| `bag_of_words` | Word-bag overlap with 0.3 floor | Fast baseline |
| `f1` | Stop-word-filtered weighted F1 | General-purpose default |
| `rouge_l` | Longest common subsequence | Sequential procedures |
| `semantic` | Embedding cosine similarity | Conversational skills |
| `graph` | Concept co-location graph | Relational analysis |
| `checklist` | Compliance-critical behavior list | Regulated skills |
| `consistency` | Cross-example output variance | Overfitting detection |

---

## Eval Dataset Sources

| Source | How |
|---|---|
| `synthetic` | LLM generates N examples from the skill text |
| `golden` | Hand-crafted examples in `golden_examples/` |
| `external` | Mined from session logs (Jiuwen / Claude Code / Hermes) |
| `trajectory` | RL trajectory store |

The holdout split is never seen by the GEPA optimizer — used only in Stage 6.

---

## Regression-Aware Gate

Stage 7 rejects any candidate where any individual quality dimension regressed
below the baseline, even if the aggregate score improved. This makes *The
Regression Trap* structurally impossible.

---

## Batch Evolution

```
all skills → Level 1 Thompson scheduler → evolve_single_skill(k) → update arm → loop
```

---

## Configuration (`EvolverConfig`)

| Parameter | Default | Description |
|---|---|---|
| `skills_root` | `~/.jiuwen/skills` | Root dir for SKILL.md files |
| `iterations` | `10` | GEPA search iterations |
| `population_size` | `5` | Candidates per iteration |
| `optimizer_model` | required | LLM for GEPA reflections |
| `eval_model` | required | LLM-as-judge |
| `fitness_metric` | `"f1"` | One of the 7 metrics |
| `eval_dataset_size` | `20` | Synthetic dataset size |
| `train_ratio` | `0.60` | Training split |
| `holdout_ratio` | `0.20` | Final evaluation split |
| `max_skill_size` | `100_000` | Hard cap on skill body (chars) |
| `max_prompt_growth` | `0.20` | Max size growth ratio |
| `min_improvement` | `0.01` | Minimum improvement to accept |
| `output_dir` | `./skill_evolver_output` | Where to write results |
