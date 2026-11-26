## Summary of Changes

- Describe what changed, why, and which issue/directive item it addresses.

---

## Tested-Definition & Honesty Reminder

Before checking the gate below, confirm all of the following:
- `tested` status in `registry/model_architectures.json` applies **ONLY** to architectures meeting all six criteria (`T-A`–`T-F`: official publisher weights, end-to-end execution, `>= 64` greedy tokens, recorded prompt/output, verbatim command, verified SHA-256 & byte size).
- Every number and non-trivial claim is backed by a reproducible command or committed fixture (`not measured` where unmeasured).
- Synthetic/toy tests are labeled explicitly as synthetic; opt-in real-weight tests do not fake-pass via silent toy fallbacks.
- `ZAQERIS/ZAQERIS-7B` descriptions state clearly that it is derived from Qwen/Qwen2.5-Coder-7B-Instruct.
- Zero hardware details (GPU/CPU specs, device memory amounts, compute-hour totals, HUD screenshots), zero deprecated internal codenames outside `zqn_12b`, zero personal names/team size/private paths/secrets/weights.

**Do not merge or release until the gate is complete (including non-author G4 approval).**

---

## Gate Checklist (`G1`–`G11`)

- [ ] **G1** CI is green on ALL jobs for the final PR head (Python tests, C++20 ubuntu/windows/macos builds, ASan+UBSan): `<links>`
- [ ] **G2** Full pytest suite passes (`python -m pytest tests/ -v`, 0 failures; only opt-in real-weight skip allowed): `<summary line>`
- [ ] **G3** ASan+UBSan run is clean (zero findings), exercising every `.c` file under `core/inference/`: `<job link>`
- [ ] **G4** Second non-author reviewer written approval on this PR (self-approval does not count): `<review/comment link>`
- [ ] **G5** Every box in this checklist is ticked with concrete evidence above and below.
- [ ] **G6** Claim-audit table attached below mapping every number/claim to a command or committed fixture (`not measured` if unmeasured).
- [ ] **G7** Release notes (if applicable) list wins AND limitations honestly with zero unsupported marketing terms.
- [ ] **G8** Semver tag target and `CHANGELOG.md` entry match cleanly.
- [ ] **G9** No force-push and no history rewrite on `main`, this PR branch, or any existing tag.
- [ ] **G10** Forbidden-content & Markdown link scans pass (`0` violations, `0` broken links): `<command outputs>`
- [ ] **G11** All scope/cleanup acceptance items completed and documented.

### G6 Claim-Audit Table

| Claim / Number | Verified Value | Reproducible Command or Committed Fixture |
| :--- | :--- | :--- |
| `...` | `...` | `...` |
