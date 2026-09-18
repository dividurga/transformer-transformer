# Working with this repo

This repo hosts an undergraduate thesis on morphology-controller co-design. Current work is Aim 0 (robustness to unmodeled physics mismatch) — full context, verified technical findings, and the step-by-step implementation plan live in `docs/aim0_plan.md`. Read that before starting or resuming Aim 0 work; its "Progress log" section at the end tracks what's actually been done vs. still open.

## Collaboration style

The author is building understanding of this codebase and the research methodology while working through it, not just directing implementation. When working through `docs/aim0_plan.md` (or similarly-scoped multi-step work):

- Work through the plan's own step boundaries one at a time — don't build ahead autonomously across steps.
- Explain what code/commands do and why, before or alongside writing them.
- Add a verification/check at the end of each step (run it, inspect real output) and confirm the result together before moving to the next step.
- Prefer small, readable diffs over large generated files dropped all at once.

Speed up or batch steps only if explicitly asked to.
