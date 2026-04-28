# mujoco_rl_fuzz_2.0

GzFuzz-style RL fuzzer for the MuJoCo physics engine — v2.0 redesign.

This directory is **isolated** from `../mujoco_rl_fuzz/` (v1). Nothing here imports
from v1; only `src/triage/signature.py` is a hardened copy of the v1 triage hash.

## Current scope (v2.0-α)

α phase implements ONLY:

1. **Seed collection** from 7 upstream repos via `tools/fetch_seeds.py`
   (shallow git clone; no submodule). Output goes to `seeds/curated/`.
2. **22 high-semantic-level mutators** (`src/mutations/ops_*.py`) keyed by
   an `IntensityTable`. No random-string field generation; everything
   queries a fixed table. lxml backend (MjSpec in MuJoCo 3.2.3 is partial).
3. **Double-layer validity gate**:
   - per-mutator post-apply `mj_loadXML` check; failure → rollback.
   - `tools/validate_mutators.py` CI gate: full
     (seed × mutator × intensity) cartesian must pass for "legal"
     intensity modes; whitelisted `invalid_parseable` modes go through
     a separate runtime-only test.

RL training, episodic runner, and oracles are **deferred to v2.0-β**.

## Layout

```
configs/default.yaml      # seed/mutator/validity config
seeds/curated/<source>/   # populated by fetch_seeds.py
seeds/_quarantine/        # MJCFs that fail mj_loadXML
src/
  mjcf/                   # XML loading + invariants + safe compile
  mutations/              # base + intensity + 6 ops_*.py files
  triage/                 # signature.py (copied from v1)
tools/                    # fetch / validate scripts
tests/                    # pytest
```

## Quickstart

```powershell
pip install -e .
python tools/fetch_seeds.py --only mujoco,dm_control
python tools/validate_seeds.py
python tools/validate_mutators.py --seed seeds/curated/mujoco__pendulum.xml
pytest -q
```
