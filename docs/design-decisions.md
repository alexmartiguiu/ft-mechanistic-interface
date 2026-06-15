# Design decisions

Open questions to resolve, with the reasoning settled so far. Terse by intent.

## LLM roles: one model for generation + judging?

Three roles, not two: (1) artifact author, (2) response producer = the **base model
under study** (Qwen), (3) judge. The judge scores (2), not its own text.

- **Not a problem for generation.** The judge grades the base model's responses, not
  its own — so "LLM grades its own homework" / self-enhancement bias doesn't apply.
  Judge ≠ producer is the separation that matters, and it holds automatically (this is
  also what Chen does: GPT-4.1-mini judges the instruct model's outputs).
- **Check sooner — author (1) and judge (3) sharing priors.** Same model defines what
  the trait *looks like* (rubric/questions) and *scores* it → mild circularity. Cheap
  mitigations: cross-vendor judge (different family from the author) + a small
  judge–human agreement spot-check (`vector-steering.md` §5). `fit_vector` already
  takes `judge` as a separate injected arg, so this is a config swap, not a refactor.
