# QA Reproducibility Audit — Dueling DQN Stock Trading

- **Date:** 2026-05-11
- **Operator:** qa-engineer subagent
- **Scope:** Task 6.1 — reproducibility / artifact / CLI audit prior to README finalization
- **Branch:** main @ 96acb51

## Results

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Test suite (project venv) | PASS | `151 passed in 10.11s` |
| 2 | Fresh venv (`python3.12 -m venv`) install + tests | PASS | clean install from `requirements.txt`; `151 passed in 40.16s` |
| 3 | Tier-3 (offline CSV fallback) coverage | PASS | `test_tier3_csv_fallback_on_yfinance_failure`, `test_tier3_missing_csv_raises_data_unavailable` — both green |
| 4 | Seeded reproducibility (NVDA, seed=42, 500 steps, x2) | PASS | MD5 of `*_latest.pt` matched bit-exactly across runs (`c7dc1b86191497d61c22b1bca03aa7f9`) — bit-exact even on MPS for this 500-step run |
| 5 | Artifact inventory | PASS | All 9 final ckpts (3.4 MB each), 9 TB event files, 3 summaries, 3 aggregates, 3 equity plots, 3 input CSVs, 6 GUI screenshots present and non-empty |
| 6 | CLI `--help` smoke (6 scripts) | PASS | All 6 scripts exit 0 with valid usage lines |
| 7 | Repo cleanliness | PASS | `working tree clean`; no stray source files |

## Notes

- **Reproducibility — better than expected.** Two 500-step trainings on `NVDA seed=42` on the local MPS backend produced byte-identical `*_latest.pt` files. The CPU-only rigorous test (`test_runner.py::test_train_one_seed_seeded_reproducible`) also passes. Strict seed discipline is honored end-to-end.
- **No real training artifacts were touched.** Audit used `seed=42`; the production `seed{0,1,2}` directories under `output/models/NVDA/` are untouched. Audit-generated `seed42` directory was cleaned up.
- **Fresh venv install** pulled the pinned dependency set with no version drift warnings; runtime was ~40s for the full 151-test suite (vs 10s in the warm project venv — within expectations).
- **Tier 3 (offline CSV fallback)** is covered by unit tests; no separate end-to-end run was needed.

## Verdict

**Ready for README finalization.** No blockers.
