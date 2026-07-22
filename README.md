# SkillForge

**Offline GEPA evolutionary optimizer for Jiuwen SKILL.md files.**

SkillForge takes a skill (a SKILL.md file) and a test dataset, runs a GEPA
population-based search to evolve the skill body, and only deploys the evolved
version if it clears a regression-aware holdout gate. It is one half of the
skill-management system; the live session reviewer lives in the companion
**SkillTend** project.

---

## Key Concepts

**GEPA (Genetic Evolutionary Prompt Adaptation)** — population-based search
over skill body text, powered by DSPy. Instead of a single gradient-like
update, GEPA maintains a population of candidate variants, evaluates each
against a fitness metric, and uses a reflection LLM to propose mutations guided
by relative scores.

**Thompson Sampling at 3 levels:**
1. **Skill scheduler** — Beta-Bernoulli bandit that concentrates optimization
   budget on skills that have previously improved rather than stalled.
2. **Training example selector** — selects the most discriminating training
   examples from the dataset to drive each GEPA mutation cycle.
3. **Bayesian acceptance gate** — requires P(evolved ≥ baseline) ≥ 0.75 before
   writing the new skill to disk.

**7 fitness metrics** — alternatives, not a combined pipeline. Pick the one
appropriate for the target skill class:
`bag_of_words` · `f1` · `rouge_l` · `semantic` · `graph` · `checklist` · `consistency`

**Regression-aware holdout gate** — no evolved skill deploys if any individual
quality dimension has regressed, regardless of aggregate score improvement.

---

## Quick Start

**CLI:**

```bash
pip install -e .

# Evolve a single skill (synthetic dataset, default fitness metric)
skillforge --skill git-review

# Evolve with custom settings
skillforge --skill git-review \
    --iterations 20 \
    --optimizer-model openai/gpt-4o \
    --eval-model openai/gpt-4o-mini \
    --fitness-metric semantic

# Evolve all skills in one pass
skillforge --all --min-improvement 0.05

# Dry run (no LLM calls — validates setup only)
skillforge --skill git-review --dry-run
```

**Programmatic:**

```python
from pathlib import Path
from skillforge.evolvers.skill_evolver_config import EvolverConfig
from skillforge.evolvers import evolve_single_skill
from skillforge.evolvers.skill_evolver_prereqs import build_evolution_prereqs
from skillforge.evolvers.skill_evolver_single_params import SkillEvolverParams

config = EvolverConfig(
    skills_root=Path.home() / ".jiuwen" / "skills",
    iterations=10,
    optimizer_model="openai/gpt-4o",
    eval_model="openai/gpt-4o-mini",
)

prereqs = build_evolution_prereqs("git-review", config, eval_source="synthetic")

params = SkillEvolverParams(
    skill_name="git-review",
    eval_source="synthetic",
    config=config,
    **prereqs.__dict__,
)

metrics = evolve_single_skill(params)
print(f"Improvement: {metrics['improvement']:+.4f}")
```

---

## Install

```bash
pip install -e .

# For benchmark examples (HuggingFace scenarios + plotting):
pip install -e ".[examples]"
```

DSPy (`>=3.0.0`) is required. It manages the GEPA optimizer and handles LLM
calls internally via LiteLLM.

---

## Project Layout

```
skillforge/
  cli.py                    # Click CLI — entry point for `python -m skillforge`
  evolvers/
    skill_evolver_config.py # EvolverConfig dataclass
    skill_evolver_single.py # evolve_single_skill() — runs the 9-stage pipeline
    skill_evolver_batch.py  # evolve_skills_batch() — runs all skills in sequence
    skill_evolver_prereqs.py# build_evolution_prereqs() — pre-flight builder
    batch_selection/        # Thompson Sampling skill scheduler (Level 1)
      skills_schedulers/
        thompson.py         # Beta-Bernoulli bandit arm per skill
        round_robin.py      # Fallback: round-robin schedule
    stages/
      stage01_skill_finder_and_loader/    # Load SKILL.md + prior metrics
      stage02_skill_constraint_validator/ # Size + frontmatter constraints
      stage03_dataset_builder/           # synthetic | golden | external | trajectory
      stage04_dspy_configurator/         # DSPy LM init + train/val/holdout split
      stage05_gepa_optimizer/            # GEPA population search (core)
      stage06_holdout_evaluator/         # 7-metric holdout scoring
      stage07_acceptance_gates/          # Thompson gate + threshold gate
      stage08_results_display/           # Rich terminal output
      stage09_output_saver/              # evolved_skill.md + metrics.json
  dataset_builder/          # Dataset construction utilities
  external_importers/       # Session log importers (Jiuwen, Claude Code, Hermes)
  skills/                   # Skill file utilities (list, load)
examples/
  other/                    # Standalone runnable examples
    offline_01_gepa_evolve.py
    offline_02_dataset_from_external.py
    offline_03_cli_dry_run_and_caching.py
  benchmarks/               # Full benchmark suite
    data/                   # 14 scenarios (synthetic + HuggingFace)
    demo/                   # Multi-run Thompson Sampling comparison demo
    fitness_metrics_oracle/ # Fitness metric oracle trainer
    skill_recommender/      # Offline skill recommender runner
docs/
  skillforge_paper.md       # Full architectural specification paper
  related_papers.md         # Literature review / related work
  skill_recommender.md      # Skill recommender system design
  architecture.md           # Architecture overview (this project)
```

---

## Relationship to SkillTend

| | SkillForge (this project) | SkillTend |
|---|---|---|
| Track | Offline — runs between sessions | Online — runs during live sessions |
| Mechanism | GEPA evolutionary optimizer | LLM-driven background review of conversation |
| Trigger | Manual CLI or batch scheduler | After every N tool-calls or M user turns |
| Writes | Whole-body SKILL.md rewrites | Targeted patches + memory entries |
| Shared | `~/.jiuwen/skills/` (SKILL.md files) | same |
| State aware | Yes — reads `metrics.json`, Thompson Sampling state | Yes — reads `.usage.json`, ACTIVE/STALE/ARCHIVED |
